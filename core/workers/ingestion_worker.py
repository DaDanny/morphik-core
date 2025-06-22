import asyncio
import contextlib
import json
import logging
import os
import time
import urllib.parse as up
from datetime import UTC, datetime
from typing import Any, Dict, List, Optional

from arq.connections import RedisSettings
from sqlalchemy import text

from core.config import get_settings
from core.database.postgres_database import PostgresDatabase
from core.embedding.colpali_api_embedding_model import ColpaliApiEmbeddingModel
from core.embedding.colpali_embedding_model import ColpaliEmbeddingModel
from core.embedding.litellm_embedding import LiteLLMEmbeddingModel
from core.limits_utils import check_and_increment_limits, estimate_pages_by_chars
from core.models.auth import AuthContext, EntityType
from core.models.rules import MetadataExtractionRule
from core.parser.morphik_parser import MorphikParser
from core.services.document_service import DocumentService
from core.services.rules_processor import RulesProcessor
from core.services.telemetry import TelemetryService
from core.storage.local_storage import LocalStorage
from core.storage.s3_storage import S3Storage
from core.vector_store.multi_vector_store import MultiVectorStore
from core.vector_store.pgvector_store import PGVectorStore

# Enterprise routing helpers
from ee.db_router import get_database_for_app, get_vector_store_for_app

logger = logging.getLogger(__name__)

# Initialize global settings once
settings = get_settings()

# Create logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)

# Set up file handler for worker_ingestion.log
file_handler = logging.FileHandler("logs/worker_ingestion.log")
file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
logger.addHandler(file_handler)
# Set logger level based on settings (diff used INFO directly)
logger.setLevel(logging.INFO)


async def get_document_with_retry(document_service, document_id, auth, max_retries=3, initial_delay=0.3):
    """
    Helper function to get a document with retries to handle race conditions.

    Args:
        document_service: The document service instance
        document_id: ID of the document to retrieve
        auth: Authentication context
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay before first attempt in seconds

    Returns:
        Document if found and accessible, None otherwise
    """
    attempt = 0
    retry_delay = initial_delay

    # Add initial delay to allow transaction to commit
    if initial_delay > 0:
        await asyncio.sleep(initial_delay)

    while attempt < max_retries:
        try:
            doc = await document_service.db.get_document(document_id, auth)
            if doc:
                logger.debug(f"Successfully retrieved document {document_id} on attempt {attempt+1}")
                return doc

            # Document not found but no exception raised
            attempt += 1
            if attempt < max_retries:
                logger.warning(
                    f"Document {document_id} not found on attempt {attempt}/{max_retries}. "
                    f"Retrying in {retry_delay}s..."
                )
                await asyncio.sleep(retry_delay)
                retry_delay *= 1.5

        except Exception as e:
            attempt += 1
            error_msg = str(e)
            if attempt < max_retries:
                logger.warning(
                    f"Error retrieving document on attempt {attempt}/{max_retries}: {error_msg}. "
                    f"Retrying in {retry_delay}s..."
                )
                await asyncio.sleep(retry_delay)
                retry_delay *= 1.5
            else:
                logger.error(f"Failed to retrieve document after {max_retries} attempts: {error_msg}")
                return None

    return None


# ---------------------------------------------------------------------------
# Profiling helpers (worker-level)
# ---------------------------------------------------------------------------

if os.getenv("ENABLE_PROFILING") == "1":
    try:
        import yappi  # type: ignore
    except ImportError:
        yappi = None
else:
    yappi = None


@contextlib.asynccontextmanager
async def _profile_ctx(label: str):  # type: ignore
    if yappi is None:
        yield
        return

    yappi.clear_stats()
    yappi.set_clock_type("cpu")
    yappi.start()
    t0 = time.perf_counter()
    try:
        yield
    finally:
        duration = time.perf_counter() - t0
        fname = f"logs/worker_{label}_{int(t0)}.prof"
        yappi.stop()
        try:
            yappi.get_func_stats().save(fname, type="pstat")
            logger.info("Saved worker profile %s (%.2fs) to %s", label, duration, fname)
        except Exception as exc:
            logger.warning("Could not save worker profile: %s", exc)


async def process_ingestion_job(
    ctx: Dict[str, Any],
    document_id: str,
    file_key: str,
    bucket: str,
    original_filename: str,
    content_type: str,
    metadata_json: str,
    auth_dict: Dict[str, Any],
    rules_list: List[Dict[str, Any]],
    use_colpali: bool,
    folder_name: Optional[str] = None,
    end_user_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Process ingestion job with text sanitization to handle PostgreSQL compatibility."""
    
    # Import sanitization utilities
    from core.utils.text_sanitization import sanitize_text_for_db, sanitize_metadata_for_db

    """
    Background worker task that processes file ingestion jobs.

    Args:
        ctx: The ARQ context dictionary
        file_key: The storage key where the file is stored
        bucket: The storage bucket name
        original_filename: The original file name
        content_type: The file's content type/MIME type
        metadata_json: JSON string of metadata
        auth_dict: Dict representation of AuthContext
        rules_list: List of rules to apply (already converted to dictionaries)
        use_colpali: Whether to use ColPali embedding model
        folder_name: Optional folder to scope the document to
        end_user_id: Optional end-user ID to scope the document to

    Returns:
        A dictionary with the document ID and processing status
    """
    try:
        # Start performance timer
        job_start_time = time.time()
        phase_times = {}
        
        # 1. Log the start of the job
        logger.info("="*80)
        logger.info(f"🚀 STARTING INGESTION JOB")
        logger.info(f"📄 File: {original_filename}")
        logger.info(f"📦 Document ID: {document_id}")
        logger.info(f"🔗 Bucket: {bucket}")
        logger.info(f"🔑 File Key: {file_key}")
        logger.info(f"📋 Content Type: {content_type}")
        logger.info(f"🎯 Use ColPali: {use_colpali}")
        logger.info(f"📁 Folder: {folder_name}")
        logger.info(f"👤 End User ID: {end_user_id}")
        logger.info("="*80)

        # 2. Deserialize metadata and auth
        logger.info("🔧 PHASE 1: Deserializing metadata and auth context...")
        deserialize_start = time.time()
        metadata = json.loads(metadata_json) if metadata_json else {}
        auth = AuthContext(
            entity_type=EntityType(auth_dict.get("entity_type", "unknown")),
            entity_id=auth_dict.get("entity_id", ""),
            app_id=auth_dict.get("app_id"),
            permissions=set(auth_dict.get("permissions", ["read"])),
            user_id=auth_dict.get("user_id", auth_dict.get("entity_id", "")),
        )
        deserialize_time = time.time() - deserialize_start
        phase_times["deserialize_auth"] = deserialize_time
        logger.info(f"✅ PHASE 1 COMPLETE: Auth deserialization took {deserialize_time:.3f}s")
        logger.info(f"   📊 Metadata keys: {list(metadata.keys()) if metadata else 'None'}")
        logger.info(f"   🔐 Auth - Entity: {auth.entity_type}, App ID: {auth.app_id}, User: {auth.user_id}")

        # ------------------------------------------------------------------
        # Per-app routing for database and vector store
        # ------------------------------------------------------------------
        logger.info("🗃️ PHASE 2: Initializing database and vector store...")
        db_init_start = time.time()

        # Resolve a dedicated database/vector-store using the JWT *app_id*.
        # When app_id is None we fall back to the control-plane resources.
        logger.info(f"   🔍 Getting database for app_id: {auth.app_id}")
        database = await get_database_for_app(auth.app_id)
        
        logger.info("   🔄 Initializing database connection...")
        await database.initialize()
        logger.info("   ✅ Database initialized successfully")

        logger.info(f"   🔍 Getting vector store for app_id: {auth.app_id}")
        vector_store = await get_vector_store_for_app(auth.app_id)
        if vector_store and hasattr(vector_store, "initialize"):
            # PGVectorStore.initialize is *async*
            try:
                logger.info("   🔄 Initializing vector store...")
                await vector_store.initialize()
                logger.info("   ✅ Vector store initialized successfully")
            except Exception as init_err:
                logger.warning(f"   ⚠️ Vector store initialization failed for app {auth.app_id}: {init_err}")
        
        db_init_time = time.time() - db_init_start
        phase_times["database_vector_store_init"] = db_init_time
        logger.info(f"✅ PHASE 2 COMPLETE: Database and vector store initialization took {db_init_time:.3f}s")

        # Initialise a per-app MultiVectorStore for ColPali when needed
        logger.info("🎨 PHASE 3: Setting up ColPali and DocumentService...")
        colpali_setup_start = time.time()
        
        colpali_vector_store = None
        if use_colpali:
            logger.info("   🔄 Initializing ColPali MultiVectorStore...")
            try:
                # Use render_as_string(hide_password=False) so the URI keeps the
                # password – str(engine.url) masks it with "***" which breaks
                # authentication for psycopg.  Also append sslmode=require when
                # missing to satisfy Neon.
                from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

                uri_raw = database.engine.url.render_as_string(hide_password=False)

                parsed = urlparse(uri_raw)
                query = parse_qs(parsed.query)
                if "sslmode" not in query and settings.MODE == "cloud":
                    query["sslmode"] = ["require"]
                    parsed = parsed._replace(query=urlencode(query, doseq=True))

                uri_final = urlunparse(parsed)

                colpali_vector_store = MultiVectorStore(uri=uri_final)
                await asyncio.to_thread(colpali_vector_store.initialize)
                logger.info("   ✅ ColPali MultiVectorStore initialized successfully")
            except Exception as e:
                logger.warning(f"   ⚠️ Failed to initialise ColPali MultiVectorStore for app {auth.app_id}: {e}")
        else:
            logger.info("   ⏩ Skipping ColPali setup (use_colpali=False)")

        # Build a fresh DocumentService scoped to this job/app so we don't
        # mutate the shared instance kept in *ctx* (avoids cross-talk between
        # concurrent jobs for different apps).
        logger.info("   🔄 Creating DocumentService instance...")
        document_service = DocumentService(
            storage=ctx["storage"],
            database=database,
            vector_store=vector_store,
            embedding_model=ctx["embedding_model"],
            parser=ctx["parser"],
            cache_factory=None,
            enable_colpali=use_colpali,
            colpali_embedding_model=ctx.get("colpali_embedding_model"),
            colpali_vector_store=colpali_vector_store,
        )
        
        colpali_setup_time = time.time() - colpali_setup_start
        phase_times["colpali_docservice_setup"] = colpali_setup_time
        logger.info(f"✅ PHASE 3 COMPLETE: ColPali and DocumentService setup took {colpali_setup_time:.3f}s")

        # 3. Download the file from storage
        logger.info("📥 PHASE 4: Downloading file from storage...")
        logger.info(f"   📂 Bucket: {bucket}")
        logger.info(f"   🔑 Key: {file_key}")
        download_start = time.time()
        
        file_content = await document_service.storage.download_file(bucket, file_key)

        # Ensure file_content is bytes
        if hasattr(file_content, "read"):
            file_content = file_content.read()
        
        download_time = time.time() - download_start
        phase_times["download_file"] = download_time
        file_size_mb = len(file_content) / 1024 / 1024
        logger.info(f"✅ PHASE 4 COMPLETE: File download took {download_time:.3f}s for {file_size_mb:.2f}MB")
        logger.info(f"   📊 Download speed: {file_size_mb/download_time:.2f} MB/s" if download_time > 0 else "   📊 Download speed: instant")

        # 4. Parse file to text
        logger.info("📝 PHASE 5: Parsing file to extract text...")
        # Use the filename derived from the storage key so the parser
        # receives the correct extension (.txt, .pdf, etc.).  Passing the UI
        # provided original_filename (often .pdf) can mislead the parser when
        # the stored object is a pre-extracted text file (e.g. .pdf.txt).
        parse_filename = os.path.basename(file_key) if file_key else original_filename
        logger.info(f"   📄 Parse filename: {parse_filename}")
        logger.info(f"   🔍 Content type: {content_type}")
        logger.info(f"   📏 File size: {len(file_content)} bytes")

        parse_start = time.time()
        additional_metadata, text = await document_service.parser.parse_file_to_text(file_content, parse_filename)
        
        parse_time = time.time() - parse_start
        phase_times["parse_file"] = parse_time
        
        text_chars = len(text)
        text_kb = text_chars / 1024
        logger.info(f"✅ PHASE 5 COMPLETE: File parsing took {parse_time:.3f}s")
        logger.info(f"   📊 Extracted text: {text_chars:,} characters ({text_kb:.1f} KB)")
        logger.info(f"   🔍 Additional metadata keys: {list(additional_metadata.keys()) if additional_metadata else 'None'}")
        if parse_time > 0:
            logger.info(f"   ⚡ Parse speed: {text_kb/parse_time:.1f} KB/s")

        # NEW -----------------------------------------------------------------
        logger.info("📏 PHASE 6: Estimating pages and checking limits...")
        limits_start = time.time()
        
        # Estimate pages early for pre-check
        num_pages_estimated = estimate_pages_by_chars(len(text))
        logger.info(f"   📖 Estimated pages: {num_pages_estimated}")

        # 4.b Enforce tier limits (pages ingested) for cloud/free tier users
        if settings.MODE == "cloud" and auth.user_id:
            logger.info(f"   🔍 Checking tier limits for user: {auth.user_id}")
            # Calculate approximate pages using same heuristic as DocumentService
            try:
                # Dry-run verification before heavy processing
                await check_and_increment_limits(
                    auth,
                    "ingest",
                    num_pages_estimated,
                    document_id,
                    verify_only=True,
                )
                logger.info("   ✅ User within tier limits")
            except Exception as limit_exc:
                logger.error(f"   ❌ User {auth.user_id} exceeded ingest limits: {limit_exc}")
                raise
        else:
            logger.info("   ⏩ Skipping limits check (not cloud mode or no user_id)")
        
        limits_time = time.time() - limits_start
        phase_times["limits_check"] = limits_time
        logger.info(f"✅ PHASE 6 COMPLETE: Limits check took {limits_time:.3f}s")
        # ---------------------------------------------------------------------

        # === Apply post_parsing rules ===
        logger.info("🔧 PHASE 7: Applying post-parsing rules...")
        rules_start = time.time()
        document_rule_metadata = {}
        if rules_list:
            logger.info(f"   📋 Processing {len(rules_list)} post-parsing rules...")
            logger.info(f"   📏 Text length before rules: {len(text):,} characters")
            
            document_rule_metadata, text = await document_service.rules_processor.process_document_rules(
                text, rules_list
            )
            metadata.update(document_rule_metadata)  # Merge metadata into main doc metadata
            
            logger.info(f"   📊 Extracted metadata keys: {list(document_rule_metadata.keys()) if document_rule_metadata else 'None'}")
            logger.info(f"   📏 Text length after rules: {len(text):,} characters")
            logger.info(f"   🔗 Total metadata keys: {list(metadata.keys()) if metadata else 'None'}")
        else:
            logger.info("   ⏩ No post-parsing rules to apply")
            
        rules_time = time.time() - rules_start
        phase_times["apply_post_parsing_rules"] = rules_time
        logger.info(f"✅ PHASE 7 COMPLETE: Post-parsing rules processing took {rules_time:.3f}s")

        # 6. Retrieve the existing document
        logger.info("📄 PHASE 8: Retrieving existing document from database...")
        retrieve_start = time.time()
        logger.info(f"   🔍 Document ID: {document_id}")
        logger.info(f"   🔐 Auth context: entity_type={auth.entity_type}, entity_id={auth.entity_id}")
        logger.info(f"   🔑 Permissions: {auth.permissions}")

        # Use the retry helper function with initial delay to handle race conditions
        logger.info("   🔄 Attempting document retrieval with retry logic...")
        doc = await get_document_with_retry(document_service, document_id, auth, max_retries=5, initial_delay=1.0)
        retrieve_time = time.time() - retrieve_start
        phase_times["retrieve_document"] = retrieve_time
        
        if not doc:
            logger.error("   ❌ Document retrieval failed!")
            logger.error(f"   📄 Document {document_id} not found in database after multiple retries")
            logger.error(f"   📂 File details - name: {original_filename}, type: {content_type}")
            logger.error(f"   🗃️ Storage - bucket: {bucket}, key: {file_key}")
            logger.error(f"   🔐 Auth - entity: {auth.entity_type}, id: {auth.entity_id}, perms: {auth.permissions}")
            raise ValueError(f"Document {document_id} not found in database after multiple retries")
        
        logger.info(f"✅ PHASE 8 COMPLETE: Document retrieval took {retrieve_time:.3f}s")
        logger.info(f"   📊 Document external_id: {doc.external_id}")
        logger.info(f"   🏷️ Document status: {doc.system_metadata.get('status', 'unknown')}")
        logger.info(f"   📋 Existing metadata keys: {list(doc.metadata.keys()) if doc.metadata else 'None'}")

        # Sanitize the extracted text
        logger.info("🧹 PHASE 9: Sanitizing text and updating document...")
        sanitize_start = time.time()
        
        original_text_len = len(text)
        sanitized_text = sanitize_text_for_db(text)
        removed_chars = original_text_len - len(sanitized_text)
        logger.info(f"   🧹 Text sanitization: removed {removed_chars} problematic characters")
        if removed_chars > 0:
            logger.info(f"   📊 Sanitized text: {original_text_len:,} → {len(sanitized_text):,} characters")

        # Prepare updates for the document
        logger.info("   🔄 Preparing document metadata updates...")
        # Merge new metadata with existing metadata to preserve external_id
        merged_metadata = {**doc.metadata, **metadata}
        # Make sure external_id is preserved in the metadata
        merged_metadata["external_id"] = doc.external_id
        logger.info(f"   📋 Merged metadata keys: {list(merged_metadata.keys())}")

        # Sanitize all metadata before storing in database
        logger.info("   🧹 Sanitizing all metadata objects...")
        sanitized_merged_metadata = sanitize_metadata_for_db(merged_metadata)
        sanitized_additional_metadata = sanitize_metadata_for_db(additional_metadata)
        
        # Create sanitized system_metadata
        sanitized_system_metadata = sanitize_metadata_for_db({**doc.system_metadata, "content": sanitized_text})

        updates = {
            "metadata": sanitized_merged_metadata,
            "additional_metadata": sanitized_additional_metadata,
            "system_metadata": sanitized_system_metadata,
        }

        # Add folder_name and end_user_id to system_metadata if provided
        if folder_name:
            updates["system_metadata"]["folder_name"] = sanitize_text_for_db(folder_name)
            logger.info(f"   📁 Added folder_name: {folder_name}")
        if end_user_id:
            updates["system_metadata"]["end_user_id"] = sanitize_text_for_db(end_user_id)
            logger.info(f"   👤 Added end_user_id: {end_user_id}")

        # Update the document in the database
        logger.info("   💾 Updating document in database...")
        update_start = time.time()
        success = await document_service.db.update_document(document_id=document_id, updates=updates, auth=auth)
        update_time = time.time() - update_start
        phase_times["update_document_parsed"] = update_time
        
        if not success:
            logger.error("   ❌ Document update failed!")
            raise ValueError(f"Failed to update document {document_id}")

        # Refresh document object with updated data
        logger.info("   🔄 Refreshing document object...")
        doc = await document_service.db.get_document(document_id, auth)
        
        sanitize_total_time = time.time() - sanitize_start
        phase_times["sanitize_and_update"] = sanitize_total_time
        logger.info(f"✅ PHASE 9 COMPLETE: Sanitization and document update took {sanitize_total_time:.3f}s")
        logger.info(f"   📊 Database update took {update_time:.3f}s")

        # 7. Split text into chunks
        logger.info("✂️ PHASE 10: Splitting text into chunks...")
        logger.info(f"   📏 Input text length: {len(text):,} characters")
        logger.info(f"   ⚙️ Chunk size: {settings.CHUNK_SIZE}")
        logger.info(f"   🔗 Chunk overlap: {settings.CHUNK_OVERLAP}")
        
        chunking_start = time.time()
        logger.info("   🔄 Starting text chunking process...")
        
        parsed_chunks = await document_service.parser.split_text(text)
        
        chunking_time = time.time() - chunking_start
        phase_times["split_into_chunks"] = chunking_time
        
        if not parsed_chunks:
            # No text was extracted from the file.  In many cases (e.g. pure images)
            # we can still proceed if ColPali multivector chunks are produced later.
            # Therefore we defer the fatal check until after ColPali chunk creation.
            logger.warning(
                "   ⚠️ No text chunks extracted after parsing. Will attempt to continue "
                "and rely on image-based chunks if available."
            )
        else:
            # Log chunk statistics
            chunk_sizes = [len(chunk.content) for chunk in parsed_chunks]
            avg_chunk_size = sum(chunk_sizes) / len(chunk_sizes) if chunk_sizes else 0
            min_chunk_size = min(chunk_sizes) if chunk_sizes else 0
            max_chunk_size = max(chunk_sizes) if chunk_sizes else 0
            
            logger.info(f"✅ PHASE 10 COMPLETE: Text chunking took {chunking_time:.3f}s")
            logger.info(f"   📊 Created {len(parsed_chunks)} chunks")
            logger.info(f"   📏 Chunk sizes - avg: {avg_chunk_size:.0f}, min: {min_chunk_size}, max: {max_chunk_size}")
            logger.info(f"   ⚡ Chunking speed: {len(text)/chunking_time:.0f} chars/sec" if chunking_time > 0 else "   ⚡ Chunking speed: instant")
            
            # Warning for slow chunking
            if chunking_time > 60:  # More than 1 minute
                logger.warning(f"   🐌 SLOW CHUNKING DETECTED: {chunking_time:.1f}s is unusually slow!")
            elif chunking_time > 10:  # More than 10 seconds
                logger.warning(f"   ⏰ Chunking took longer than expected: {chunking_time:.1f}s")

        # Decide whether we need image chunks either for ColPali embedding or because
        # there are image-based rules (use_images=True) that must process them.
        logger.info("🎨 PHASE 11: Analyzing image chunk requirements and creating ColPali chunks...")
        
        has_image_rules = any(
            r.get("stage", "post_parsing") == "post_chunking"
            and r.get("type") == "metadata_extraction"
            and r.get("use_images", False)
            for r in rules_list or []
        )

        using_colpali = (
            use_colpali and document_service.colpali_embedding_model and document_service.colpali_vector_store
        )

        logger.info(f"   🎨 ColPali available: {using_colpali} turned off for testing")
        using_colpali = False

        should_create_image_chunks = has_image_rules or using_colpali
        
        logger.info(f"   🔍 Image rules detected: {has_image_rules}")
        logger.info(f"   🎨 ColPali available: {using_colpali}")
        logger.info(f"   🤔 Should create image chunks: {should_create_image_chunks}")

        # Start timer for optional image chunk creation / multivector processing
        colpali_processing_start = time.time()

        chunks_multivector = []
        if should_create_image_chunks:
            logger.info("   🔄 Creating multivector/image chunks...")
            import base64
            import filetype

            file_type = filetype.guess(file_content)
            logger.info(f"   📄 Detected file type: {file_type}")
            
            logger.info("   🔄 Encoding file content to base64...")
            file_content_base64 = base64.b64encode(file_content).decode()
            logger.info(f"   📊 Base64 encoded size: {len(file_content_base64):,} characters")

            # Use the parsed chunks for ColPali/image rules – this will create image chunks if appropriate
            logger.info("   🔄 Creating chunks using _create_chunks_multivector...")
            chunks_multivector = document_service._create_chunks_multivector(
                file_type, file_content_base64, file_content, parsed_chunks
            )
            logger.info(f"   ✅ Created {len(chunks_multivector)} multivector/image chunks")
        else:
            logger.info("   ⏩ Skipping image chunk creation (not needed)")
            
        colpali_create_chunks_time = time.time() - colpali_processing_start
        phase_times["colpali_create_chunks"] = colpali_create_chunks_time
        
        logger.info(f"✅ PHASE 11 COMPLETE: ColPali chunk creation took {colpali_create_chunks_time:.3f}s")
        if should_create_image_chunks:
            logger.info(f"   📊 Image chunks created: {len(chunks_multivector)}")
            logger.info(f"   ⚡ Creation speed: {len(chunks_multivector)/colpali_create_chunks_time:.1f} chunks/sec" if colpali_create_chunks_time > 0 else "   ⚡ Creation speed: instant")
            
            # Warning for slow ColPali chunk creation
            if colpali_create_chunks_time > 120:  # More than 2 minutes
                logger.warning(f"   🐌 SLOW COLPALI CHUNK CREATION: {colpali_create_chunks_time:.1f}s is unusually slow!")
            elif colpali_create_chunks_time > 30:  # More than 30 seconds
                logger.warning(f"   ⏰ ColPali chunk creation took longer than expected: {colpali_create_chunks_time:.1f}s")

        # If we still have no chunks at all (neither text nor image) abort early
        if not parsed_chunks and not chunks_multivector:
            raise ValueError("No content chunks (text or image) could be extracted from the document")

        # Determine the final page count for recording usage
        final_page_count = num_pages_estimated  # Default to estimate
        if using_colpali and chunks_multivector:
            final_page_count = len(chunks_multivector)
        final_page_count = max(1, final_page_count)  # Ensure at least 1 page
        logger.info(
            f"Determined final page count for usage recording: {final_page_count} pages (ColPali used: {using_colpali})"
        )

        colpali_count_for_limit_fn = len(chunks_multivector) if using_colpali else None

        # 9. Apply post_chunking rules and aggregate metadata
        logger.info("🔧 PHASE 12: Applying post-chunking rules and processing chunks...")
        post_chunking_start = time.time()
        
        processed_chunks = []
        processed_chunks_multivector = []
        aggregated_chunk_metadata: Dict[str, Any] = {}  # Initialize dict for aggregated metadata
        chunk_contents = []  # Initialize list to collect chunk contents as we process them

        if rules_list:
            logger.info(f"   📋 Found {len(rules_list)} total rules to process...")

            # Partition rules by type
            text_rules = []
            image_rules = []

            for rule_dict in rules_list:
                rule = document_service.rules_processor._parse_rule(rule_dict)
                if rule.stage == "post_chunking":
                    if isinstance(rule, MetadataExtractionRule) and rule.use_images:
                        image_rules.append(rule_dict)
                    else:
                        text_rules.append(rule_dict)

            logger.info(f"   🔍 Partitioned rules: {len(text_rules)} text rules, {len(image_rules)} image rules")

            # Process regular text chunks with text rules only
            if text_rules:
                logger.info(f"   🔄 Applying {len(text_rules)} text rules to {len(parsed_chunks)} text chunks...")
                for i, chunk_obj in enumerate(parsed_chunks, 1):
                    if i % 50 == 0:  # Log progress every 50 chunks
                        logger.info(f"   📊 Processing text chunk {i}/{len(parsed_chunks)}...")
                    # Get metadata *and* the potentially modified chunk
                    chunk_rule_metadata, processed_chunk = await document_service.rules_processor.process_chunk_rules(
                        chunk_obj, text_rules
                    )
                    processed_chunks.append(processed_chunk)
                    chunk_contents.append(processed_chunk.content)  # Collect content as we process
                    # Aggregate the metadata extracted from this chunk
                    aggregated_chunk_metadata.update(chunk_rule_metadata)
                logger.info(f"   ✅ Completed text rules processing for {len(processed_chunks)} chunks")
            else:
                logger.info("   ⏩ No text rules to apply, using original chunks")
                processed_chunks = parsed_chunks  # No text rules, use original chunks

            # Process colpali image chunks with image rules if they exist
            if chunks_multivector and image_rules:
                logger.info(f"   🔄 Applying {len(image_rules)} image rules to {len(chunks_multivector)} image chunks...")
                image_chunks_processed = 0
                for i, chunk_obj in enumerate(chunks_multivector, 1):
                    # Only process if it's an image chunk - pass the image content to the rule
                    if chunk_obj.metadata.get("is_image", False):
                        image_chunks_processed += 1
                        if image_chunks_processed % 20 == 0:  # Log progress every 20 image chunks
                            logger.info(f"   📊 Processing image chunk {image_chunks_processed}...")
                        # Get metadata *and* the potentially modified chunk
                        chunk_rule_metadata, processed_chunk = (
                            await document_service.rules_processor.process_chunk_rules(chunk_obj, image_rules)
                        )
                        processed_chunks_multivector.append(processed_chunk)
                        # Aggregate the metadata extracted from this chunk
                        aggregated_chunk_metadata.update(chunk_rule_metadata)
                    else:
                        # Non-image chunks from multivector don't need further processing
                        processed_chunks_multivector.append(chunk_obj)

                logger.info(f"   ✅ Completed image rules processing for {image_chunks_processed} image chunks")
            elif chunks_multivector:
                # No image rules, use original multivector chunks
                logger.info("   ⏩ No image rules to apply, using original multivector chunks")
                processed_chunks_multivector = chunks_multivector

            logger.info(f"   📊 Final results - text chunks: {len(processed_chunks)}, multivector chunks: {len(processed_chunks_multivector)}")
            logger.info(f"   📋 Aggregated metadata keys: {list(aggregated_chunk_metadata.keys()) if aggregated_chunk_metadata else 'None'}")

            # Update the document content with the stitched content from processed chunks
            if processed_chunks:
                logger.info("   🔄 Updating document content with processed chunks...")
                stitched_content = "\n".join(chunk_contents)
                original_stitched_len = len(stitched_content)
                # Sanitize the stitched content before storing
                sanitized_stitched_content = sanitize_text_for_db(stitched_content)
                removed_chars = original_stitched_len - len(sanitized_stitched_content)
                if removed_chars > 0:
                    logger.info(f"   🧹 Sanitized stitched content: removed {removed_chars} problematic characters")
                # Apply sanitization to the entire system_metadata object
                doc.system_metadata["content"] = sanitized_stitched_content
                doc.system_metadata = sanitize_metadata_for_db(doc.system_metadata)
                logger.info(f"   ✅ Updated document content length: {len(sanitized_stitched_content):,} characters")
        else:
            logger.info("   ⏩ No post-chunking rules to apply")
            processed_chunks = parsed_chunks  # No rules, use original chunks
            processed_chunks_multivector = chunks_multivector  # No rules, use original multivector chunks
            
        post_chunking_time = time.time() - post_chunking_start
        phase_times["apply_post_chunking_rules"] = post_chunking_time
        logger.info(f"✅ PHASE 12 COMPLETE: Post-chunking rules processing took {post_chunking_time:.3f}s")
        
        # Warning for slow post-chunking rules
        if post_chunking_time > 300:  # More than 5 minutes
            logger.warning(f"   🐌 SLOW POST-CHUNKING RULES: {post_chunking_time:.1f}s is unusually slow!")
        elif post_chunking_time > 60:  # More than 1 minute
            logger.warning(f"   ⏰ Post-chunking rules took longer than expected: {post_chunking_time:.1f}s")

        # 10. Generate embeddings for processed chunks
        logger.info("🧠 PHASE 13: Generating embeddings for text chunks...")
        logger.info(f"   📊 Chunks to embed: {len(processed_chunks)}")
        logger.info(f"   🔗 Embedding model: {settings.EMBEDDING_MODEL}")
        
        embedding_start = time.time()
        logger.info("   🔄 Starting embedding generation...")
        
        embeddings = await document_service.embedding_model.embed_for_ingestion(processed_chunks)
        
        embedding_time = time.time() - embedding_start
        phase_times["generate_embeddings"] = embedding_time
        embeddings_per_second = len(embeddings) / embedding_time if embedding_time > 0 else 0
        
        logger.info(f"✅ PHASE 13 COMPLETE: Embedding generation took {embedding_time:.3f}s")
        logger.info(f"   📊 Generated {len(embeddings)} embeddings")
        logger.info(f"   ⚡ Embedding speed: {embeddings_per_second:.2f} embeddings/sec")
        
        # Warning for slow embedding
        if embedding_time > 300:  # More than 5 minutes
            logger.warning(f"   🐌 SLOW EMBEDDING GENERATION: {embedding_time:.1f}s is unusually slow!")
        elif embedding_time > 60:  # More than 1 minute
            logger.warning(f"   ⏰ Embedding generation took longer than expected: {embedding_time:.1f}s")

        # 11. Create chunk objects with potentially modified chunk content and metadata
        logger.info("📦 PHASE 14: Creating chunk objects...")
        logger.info(f"   📊 Chunks: {len(processed_chunks)}, Embeddings: {len(embeddings)}")
        
        chunk_objects_start = time.time()
        chunk_objects = document_service._create_chunk_objects(doc.external_id, processed_chunks, embeddings)
        chunk_objects_time = time.time() - chunk_objects_start
        phase_times["create_chunk_objects"] = chunk_objects_time
        
        logger.info(f"✅ PHASE 14 COMPLETE: Creating chunk objects took {chunk_objects_time:.3f}s")
        logger.info(f"   📦 Created {len(chunk_objects)} chunk objects")

        # 12. Handle ColPali embeddings
        logger.info("🎨 PHASE 15: Generating ColPali embeddings...")
        colpali_embed_start = time.time()
        chunk_objects_multivector = []
        
        if using_colpali:
            logger.info(f"   📊 Multivector chunks to embed: {len(processed_chunks_multivector)}")
            logger.info("   🔄 Starting ColPali embedding generation...")
            
            colpali_embeddings = await document_service.colpali_embedding_model.embed_for_ingestion(
                processed_chunks_multivector
            )
            logger.info(f"   ✅ Generated {len(colpali_embeddings)} ColPali embeddings")

            logger.info("   📦 Creating ColPali chunk objects...")
            chunk_objects_multivector = document_service._create_chunk_objects(
                doc.external_id, processed_chunks_multivector, colpali_embeddings
            )
            logger.info(f"   📦 Created {len(chunk_objects_multivector)} ColPali chunk objects")
        else:
            logger.info("   ⏩ Skipping ColPali embedding (not using ColPali)")
            
        colpali_embed_time = time.time() - colpali_embed_start
        phase_times["colpali_generate_embeddings"] = colpali_embed_time
        
        logger.info(f"✅ PHASE 15 COMPLETE: ColPali embedding took {colpali_embed_time:.3f}s")
        if using_colpali:
            embeddings_per_second = len(colpali_embeddings) / colpali_embed_time if colpali_embed_time > 0 else 0
            logger.info(f"   ⚡ ColPali embedding speed: {embeddings_per_second:.2f} embeddings/sec")
            
            # Warning for slow ColPali embedding
            if colpali_embed_time > 600:  # More than 10 minutes
                logger.warning(f"   🐌 SLOW COLPALI EMBEDDING: {colpali_embed_time:.1f}s is unusually slow!")
            elif colpali_embed_time > 120:  # More than 2 minutes
                logger.warning(f"   ⏰ ColPali embedding took longer than expected: {colpali_embed_time:.1f}s")

        # === Merge aggregated chunk metadata into document metadata ===
        if aggregated_chunk_metadata:
            logger.info("Merging aggregated chunk metadata into document metadata...")
            # Make sure doc.metadata exists
            if not hasattr(doc, "metadata") or doc.metadata is None:
                doc.metadata = {}
            # Sanitize the aggregated metadata before merging
            sanitized_aggregated_metadata = sanitize_metadata_for_db(aggregated_chunk_metadata)
            doc.metadata.update(sanitized_aggregated_metadata)
            logger.info(f"Final document metadata after merge: {doc.metadata}")
        # ===========================================================

        logger.info("💾 PHASE 16: Final document preparation and storage...")
        logger.info("   🔄 Preparing document for final storage...")
        
        # Update document status to completed before storing
        doc.system_metadata["status"] = "completed"
        doc.system_metadata["updated_at"] = datetime.now(UTC)
        
        # Sanitize system_metadata before final update
        logger.info("   🧹 Final sanitization of all metadata...")
        doc.system_metadata = sanitize_metadata_for_db(doc.system_metadata)
        # Sanitize other metadata fields as well
        doc.metadata = sanitize_metadata_for_db(doc.metadata)
        if hasattr(doc, 'additional_metadata'):
            doc.additional_metadata = sanitize_metadata_for_db(doc.additional_metadata)

        # 11. Store chunks and update document with is_update=True
        logger.info(f"   💾 Storing {len(chunk_objects)} text chunks and {len(chunk_objects_multivector)} ColPali chunks...")
        store_start = time.time()
        
        await document_service._store_chunks_and_doc(
            chunk_objects, doc, use_colpali, chunk_objects_multivector, is_update=True, auth=auth
        )
        
        store_time = time.time() - store_start
        phase_times["store_chunks_and_update_doc"] = store_time
        
        logger.info(f"✅ PHASE 16 COMPLETE: Final storage took {store_time:.3f}s")
        logger.info(f"   📊 Stored {len(chunk_objects)} text chunks")
        if chunk_objects_multivector:
            logger.info(f"   📊 Stored {len(chunk_objects_multivector)} ColPali chunks")
        
        # Warning for slow storage
        if store_time > 300:  # More than 5 minutes
            logger.warning(f"   🐌 SLOW DOCUMENT STORAGE: {store_time:.1f}s is unusually slow!")
        elif store_time > 60:  # More than 1 minute
            logger.warning(f"   ⏰ Document storage took longer than expected: {store_time:.1f}s")

        logger.info(f"🎉 Successfully completed processing for document {doc.external_id}")

        # 13. Log successful completion
        logger.info("="*80)
        logger.info(f"🎉 INGESTION COMPLETED SUCCESSFULLY!")
        logger.info(f"📄 File: {original_filename}")
        logger.info(f"📦 Document ID: {doc.external_id}")
        logger.info("="*80)
        # Performance summary
        total_time = time.time() - job_start_time

        # Log detailed performance summary
        logger.info("📊 DETAILED PERFORMANCE SUMMARY")
        logger.info("="*80)
        logger.info(f"🕐 TOTAL PROCESSING TIME: {total_time:.3f}s ({total_time/60:.1f} minutes)")
        logger.info("-"*80)
        
        # Sort phases by duration (slowest first) for easy identification of bottlenecks
        sorted_phases = sorted(phase_times.items(), key=lambda x: x[1], reverse=True)
        
        for i, (phase, duration) in enumerate(sorted_phases, 1):
            percentage = (duration / total_time) * 100 if total_time > 0 else 0
            
            # Add visual indicators for slow phases
            if percentage > 50:
                indicator = "🚨"  # Critical - over 50% of time
            elif percentage > 25:
                indicator = "⚠️"   # Warning - over 25% of time
            elif percentage > 10:
                indicator = "📍"  # Notable - over 10% of time
            else:
                indicator = "✅"  # Normal
            
            logger.info(f"{indicator} #{i:2d}. {phase:<35} {duration:8.3f}s ({percentage:5.1f}%)")
        
        logger.info("-"*80)
        logger.info(f"📊 FILE SIZE: {len(file_content)/1024/1024:.2f} MB")
        logger.info(f"📄 TEXT EXTRACTED: {len(text):,} characters")
        logger.info(f"📦 TEXT CHUNKS: {len(processed_chunks)}")
        logger.info(f"🎨 COLPALI CHUNKS: {len(chunk_objects_multivector)}")
        logger.info(f"📖 FINAL PAGE COUNT: {final_page_count}")
        logger.info(f"⚡ OVERALL SPEED: {len(text)/(total_time*1024):.1f} KB/s")
        logger.info("="*80)

        # Record ingest usage *after* successful completion using the final page count
        if settings.MODE == "cloud" and auth.user_id:
            logger.info("📊 Recording usage limits...")
            try:
                await check_and_increment_limits(
                    auth,
                    "ingest",
                    final_page_count,
                    document_id,
                    use_colpali=using_colpali,
                    colpali_chunks_count=colpali_count_for_limit_fn,
                )
                logger.info("✅ Usage limits recorded successfully")
            except Exception as rec_exc:
                logger.error(f"❌ Failed to record ingest usage after completion: {rec_exc}")
        else:
            logger.info("⏩ Skipping usage limits recording (not cloud mode)")

        # 14. Return document ID
        logger.info("🎯 RETURNING SUCCESS RESULT")
        result = {
            "document_id": document_id,
            "status": "completed",
            "filename": original_filename,
            "content_type": content_type,
            "timestamp": datetime.now(UTC).isoformat(),
            "processing_time_seconds": total_time,
            "chunks_created": len(chunk_objects),
            "colpali_chunks_created": len(chunk_objects_multivector),
            "pages_processed": final_page_count,
        }
        logger.info(f"📋 Result: {result}")
        return result

    except Exception as e:
        # Calculate elapsed time for error reporting
        error_time = time.time() - job_start_time
        logger.error("="*80)
        logger.error("❌ INGESTION JOB FAILED!")
        logger.error(f"📄 File: {original_filename}")
        logger.error(f"📦 Document ID: {document_id}")
        logger.error(f"⏱️ Failed after: {error_time:.3f}s ({error_time/60:.1f} minutes)")
        logger.error(f"🔥 Error: {str(e)}")
        logger.error("="*80)
        
        # Log phase times up to the point of failure
        if phase_times:
            logger.error("📊 PHASES COMPLETED BEFORE FAILURE:")
            for phase, duration in phase_times.items():
                percentage = (duration / error_time) * 100 if error_time > 0 else 0
                logger.error(f"   ✅ {phase}: {duration:.3f}s ({percentage:.1f}%)")
            logger.error("-"*80)

        # ------------------------------------------------------------------
        # Ensure we update the *per-app* database where the document lives.
        # Falling back to the control-plane DB (ctx["database"]) can silently
        # fail because the row doesn't exist there.
        # ------------------------------------------------------------------

        try:
            database: Optional[PostgresDatabase] = None

            # Prefer the tenant-specific database
            if auth.app_id is not None:
                try:
                    database = await get_database_for_app(auth.app_id)
                    await database.initialize()
                except Exception as db_err:
                    logger.warning(
                        "Failed to obtain per-app database in error handler: %s. Falling back to default.",
                        db_err,
                    )

            # Fallback to the default database kept in the worker context
            if database is None:
                database = ctx.get("database")

            # Proceed only if we have a database object
            if database:
                # Try to get the document
                doc = await database.get_document(document_id, auth)

                if doc:
                    # Sanitize error message before storing
                    sanitized_error = sanitize_text_for_db(str(e))
                    
                    # Update the document status to failed
                    await database.update_document(
                        document_id=document_id,
                        updates={
                            "system_metadata": sanitize_metadata_for_db({
                                **doc.system_metadata,
                                "status": "failed",
                                "error": sanitized_error,
                                "updated_at": datetime.now(UTC),
                            })
                        },
                        auth=auth,
                    )
                    logger.info(f"Updated document {document_id} status to failed")
        except Exception as inner_e:
            logger.error(f"Failed to update document status: {inner_e}")

        # Note: We don't record usage if the job failed.

        # Return error information
        return {
            "status": "failed",
            "filename": original_filename,
            "error": str(e),
            "timestamp": datetime.now(UTC).isoformat(),
        }


async def startup(ctx):
    """
    Worker startup: Initialize all necessary services that will be reused across jobs.

    This initialization is similar to what happens in core/api.py during app startup,
    but adapted for the worker context.
    """
    logger.info("Worker starting up. Initializing services...")

    # Initialize database
    logger.info("Initializing database...")
    database = PostgresDatabase(uri=settings.POSTGRES_URI)
    # database = PostgresDatabase(uri="postgresql+asyncpg://morphik:morphik@postgres:5432/morphik")
    success = await database.initialize()
    if success:
        logger.info("Database initialization successful")
    else:
        logger.error("Database initialization failed")
    ctx["database"] = database

    # Initialize vector store
    logger.info("Initializing primary vector store...")
    vector_store = PGVectorStore(uri=settings.POSTGRES_URI)
    # vector_store = PGVectorStore(uri="postgresql+asyncpg://morphik:morphik@postgres:5432/morphik")
    success = await vector_store.initialize()
    if success:
        logger.info("Primary vector store initialization successful")
    else:
        logger.error("Primary vector store initialization failed")
    ctx["vector_store"] = vector_store

    # Initialize storage
    if settings.STORAGE_PROVIDER == "local":
        storage = LocalStorage(storage_path=settings.STORAGE_PATH)
    elif settings.STORAGE_PROVIDER == "aws-s3":
        storage = S3Storage(
            aws_access_key=settings.AWS_ACCESS_KEY,
            aws_secret_key=settings.AWS_SECRET_ACCESS_KEY,
            region_name=settings.AWS_REGION,
            default_bucket=settings.S3_BUCKET,
        )
    else:
        raise ValueError(f"Unsupported storage provider: {settings.STORAGE_PROVIDER}")
    ctx["storage"] = storage

    # Initialize parser
    parser = MorphikParser(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        use_unstructured_api=settings.USE_UNSTRUCTURED_API,
        unstructured_api_key=settings.UNSTRUCTURED_API_KEY,
        assemblyai_api_key=settings.ASSEMBLYAI_API_KEY,
        anthropic_api_key=settings.ANTHROPIC_API_KEY,
        use_contextual_chunking=settings.USE_CONTEXTUAL_CHUNKING,
    )
    ctx["parser"] = parser

    # Initialize embedding model
    # Smart embedding model selection - use direct Ollama client for Ollama models to avoid LiteLLM bugs
    from core.embedding.ollama_embedding_model import OllamaEmbeddingModel
    
    embedding_model_config = settings.REGISTERED_MODELS.get(settings.EMBEDDING_MODEL, {})
    embedding_model_name = embedding_model_config.get("model_name", "")

    if "ollama" in embedding_model_name.lower():
        # Use direct Ollama embedding model to bypass LiteLLM bug
        embedding_model = OllamaEmbeddingModel(model_key=settings.EMBEDDING_MODEL)
        logger.info("Worker: Initialized direct Ollama embedding model with model key: %s", settings.EMBEDDING_MODEL)
    else:
        # Use LiteLLM for non-Ollama models
        embedding_model = LiteLLMEmbeddingModel(model_key=settings.EMBEDDING_MODEL)
        logger.info("Worker: Initialized LiteLLM embedding model with model key: %s", settings.EMBEDDING_MODEL)
    
    ctx["embedding_model"] = embedding_model

    # Skip initializing completion model and reranker since they're not needed for ingestion

    # Initialize ColPali embedding model and vector store per mode
    colpali_embedding_model = None
    colpali_vector_store = None
    should_use_colpali = False # settings.COLPALI_MODE != "off"
    if should_use_colpali:
        logger.info(f"Initializing ColPali components (mode={settings.COLPALI_MODE}) ...")
        # Choose embedding implementation
        match settings.COLPALI_MODE:
            case "local":
                colpali_embedding_model = ColpaliEmbeddingModel()
            case "api":
                colpali_embedding_model = ColpaliApiEmbeddingModel()
            case _:
                raise ValueError(f"Unsupported COLPALI_MODE: {settings.COLPALI_MODE}")

        # Vector store is needed for both local and api modes
        colpali_vector_store = MultiVectorStore(uri=settings.POSTGRES_URI)
        # colpali_vector_store = MultiVectorStore(uri="postgresql+asyncpg://morphik:morphik@postgres:5432/morphik")
        success = await asyncio.to_thread(colpali_vector_store.initialize)
        if success:
            logger.info("ColPali vector store initialization successful")
        else:
            logger.error("ColPali vector store initialization failed")
    ctx["colpali_embedding_model"] = colpali_embedding_model
    ctx["colpali_vector_store"] = colpali_vector_store
    ctx["cache_factory"] = None

    # Initialize rules processor
    rules_processor = RulesProcessor()
    ctx["rules_processor"] = rules_processor

    # Initialize telemetry service
    telemetry = TelemetryService()
    ctx["telemetry"] = telemetry

    # Create the document service using only the components needed for ingestion
    document_service = DocumentService(
        storage=storage,
        database=database,
        vector_store=vector_store,
        embedding_model=embedding_model,
        parser=parser,
        cache_factory=None,
        enable_colpali=(settings.COLPALI_MODE != "off"),
        colpali_embedding_model=colpali_embedding_model,
        colpali_vector_store=colpali_vector_store,
    )
    ctx["document_service"] = document_service

    logger.info("Worker startup complete. All services initialized.")


async def shutdown(ctx):
    """
    Worker shutdown: Clean up resources.

    Properly close connections and cleanup resources to prevent leaks.
    """
    logger.info("Worker shutting down. Cleaning up resources...")

    # Close database connections
    if "database" in ctx and hasattr(ctx["database"], "engine"):
        logger.info("Closing database connections...")
        await ctx["database"].engine.dispose()

    # Close vector store connections if they exist
    if "vector_store" in ctx and hasattr(ctx["vector_store"], "engine"):
        logger.info("Closing vector store connections...")
        await ctx["vector_store"].engine.dispose()

    # Close colpali vector store connections if they exist
    if "colpali_vector_store" in ctx and hasattr(ctx["colpali_vector_store"], "engine"):
        logger.info("Closing colpali vector store connections...")
        await ctx["colpali_vector_store"].engine.dispose()

    # Close any other open connections or resources that need cleanup
    logger.info("Worker shutdown complete.")


def redis_settings_from_env() -> RedisSettings:
    """
    Create RedisSettings from environment variables for ARQ worker.

    Returns:
        RedisSettings configured for Redis connection with optimized performance
    """
    url = up.urlparse(os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0"))

    # Use ARQ's supported parameters with optimized values for stability
    # For high-volume ingestion (100+ documents), these settings help prevent timeouts
    return RedisSettings(
        host=settings.REDIS_HOST,
        port=settings.REDIS_PORT,
        database=int(url.path.lstrip("/") or 0),
        conn_timeout=5,  # Increased connection timeout (seconds)
        conn_retries=15,  # More retries for transient connection issues
        conn_retry_delay=1,  # Quick retry delay (seconds)
    )


# ARQ Worker Settings
class WorkerSettings:
    """
    ARQ Worker settings for the ingestion worker.

    This defines the functions available to the worker, startup and shutdown handlers,
    and any specific Redis settings.
    """

    functions = [process_ingestion_job]
    on_startup = startup
    on_shutdown = shutdown

    # Use robust Redis settings that handle connection issues
    redis_settings = redis_settings_from_env()

    # Result storage settings
    keep_result_ms = 24 * 60 * 60 * 1000  # Keep results for 24 hours (24 * 60 * 60 * 1000 ms)

    # Concurrency settings - optimized for high-volume ingestion
    max_jobs = 3  # Reduced to prevent resource contention during batch processing

    # Resource management
    health_check_interval = 600  # Extended to 10 minutes to reduce Redis overhead
    job_timeout = 7200  # Extended to 2 hours for large document processing
    max_tries = 5  # Retry failed jobs up to 5 times
    poll_delay = 2.0  # Increased poll delay to prevent Redis connection saturation

    # High reliability settings
    allow_abort_jobs = False  # Don't abort jobs on worker shutdown
    retry_jobs = True  # Always retry failed jobs

    # Prevent queue blocking on error
    skip_queue_when_queues_read_fails = True  # Continue processing other queues if one fails

    # Log Redis and connection pool information for debugging
    @staticmethod
    async def health_check(ctx):
        """
        Enhanced periodic health check to log connection status and job stats.
        Monitors Redis memory, database connections, and job processing metrics.
        """
        database = ctx.get("database")
        vector_store = ctx.get("vector_store")
        job_stats = ctx.get("job_stats", {})

        # Get detailed Redis info
        try:
            redis_info = await ctx["redis"].info(section=["Server", "Memory", "Clients", "Stats"])

            # Server and resource usage info
            redis_version = redis_info.get("redis_version", "unknown")
            used_memory = redis_info.get("used_memory_human", "unknown")
            used_memory_peak = redis_info.get("used_memory_peak_human", "unknown")
            clients_connected = redis_info.get("connected_clients", "unknown")
            rejected_connections = redis_info.get("rejected_connections", 0)
            total_commands = redis_info.get("total_commands_processed", 0)

            # DB keys
            db_info = redis_info.get("db0", {})
            keys_count = db_info.get("keys", 0) if isinstance(db_info, dict) else 0

            # Log comprehensive server status
            logger.info(
                f"Redis Status: v{redis_version} | "
                f"Memory: {used_memory} (peak: {used_memory_peak}) | "
                f"Clients: {clients_connected} (rejected: {rejected_connections}) | "
                f"DB Keys: {keys_count} | Commands: {total_commands}"
            )

            # Check for memory warning thresholds
            if isinstance(used_memory, str) and used_memory.endswith("G"):
                memory_value = float(used_memory[:-1])
                if memory_value > 1.0:  # More than 1GB used
                    logger.warning(f"Redis memory usage is high: {used_memory}")

            # Check for connection issues
            if rejected_connections and int(rejected_connections) > 0:
                logger.warning(f"Redis has rejected {rejected_connections} connections")
        except Exception as e:
            logger.error(f"Failed to get Redis info: {str(e)}")

        # Log job statistics with detailed processing metrics
        ongoing = job_stats.get("ongoing", 0)
        queued = job_stats.get("queued", 0)

        logger.info(
            f"Job Stats: completed={job_stats.get('complete', 0)} | "
            f"failed={job_stats.get('failed', 0)} | "
            f"retried={job_stats.get('retried', 0)} | "
            f"ongoing={ongoing} | queued={queued}"
        )

        # Warn if too many jobs are queued/backed up
        if queued > 50:
            logger.warning(f"Large job queue backlog: {queued} jobs waiting")

        # Test database connectivity with extended timeout
        if database and hasattr(database, "async_session"):
            try:
                async with database.async_session() as session:
                    await session.execute(text("SELECT 1"))
                    logger.debug("Database connection is healthy")
            except Exception as e:
                logger.error(f"Database connection test failed: {str(e)}")

        # Test vector store connectivity if available
        if vector_store and hasattr(vector_store, "async_session"):
            try:
                async with vector_store.get_session_with_retry() as session:
                    logger.debug("Vector store connection is healthy")
            except Exception as e:
                logger.error(f"Vector store connection test failed: {str(e)}")
