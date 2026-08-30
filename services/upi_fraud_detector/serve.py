"""Serving script for UPI Fraud Detector."""

import asyncio
import grpc
from concurrent import futures
import structlog
from omegaconf import OmegaConf

from service import UpiFraudDetector

logger = structlog.get_logger(__name__)


async def serve(config: OmegaConf):
    """Start gRPC server."""
    detector = UpiFraudDetector(config)
    await detector.initialize()
    
    # Create gRPC server
    server = grpc.aio.server(
        futures.ThreadPoolExecutor(max_workers=10),
        options=[
            ('grpc.max_send_message_length', 50 * 1024 * 1024),
            ('grpc.max_receive_message_length', 50 * 1024 * 1024),
        ]
    )
    
    # Add service (would use generated gRPC stubs)
    # upi_fraud_detection_pb2_grpc.add_UpiFraudServiceServicer_to_server(
    #     UpiFraudServicer(detector), server
    # )
    
    # HTTP server for health checks
    from aiohttp import web
    
    async def health(request):
        return web.json_response({"status": "healthy", "service": "upi_fraud_detector"})
    
    async def metrics(request):
        from libs.common.metrics import get_metrics_output, get_content_type
        return web.Response(body=get_metrics_output(), content_type=get_content_type())
    
    app = web.Application()
    app.router.add_get("/health", health)
    app.router.add_get("/metrics", metrics)
    
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", 8080)
    await site.start()
    
    grpc_port = config.get("serving", {}).get("grpc_port", 50051)
    server.add_insecure_port(f"[::]:{grpc_port}")
    
    logger.info(f"Starting UPI Fraud Detector on gRPC:{grpc_port}, HTTP:8080")
    
    await server.start()
    
    try:
        await server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        await server.stop(grace=5)
        await runner.cleanup()


class UpiFraudServicer:
    """gRPC servicer for UPI fraud detection."""
    
    def __init__(self, detector: UpiFraudDetector):
        self.detector = detector
    
    async def ScoreUpiTxn(self, request, context):
        """Score a single UPI transaction."""
        from libs.common.schemas import UpiTxnRequest
        
        # Convert gRPC request to internal schema
        internal_request = UpiTxnRequest(
            request_id=request.request_id,
            merchant_id=request.merchant_id,
            txn_id=request.txn_id,
            amount_paise=request.amount_paise,
            upi_handle=request.upi_handle,
            device_fingerprint=request.device_fingerprint,
            qr_image_b64=request.qr_image_b64,
            screenshot_b64=request.screenshot_b64,
            screenshot_ts_ms=request.screenshot_ts_ms,
        )
        
        # Score transaction
        response = await self.detector.score_transaction(internal_request)
        
        # Convert to gRPC response
        # return upi_fraud_detection_pb2.UpiTxnResponse(...)
        pass
    
    async def BatchScoreUpiTxns(self, request_iterator, context):
        """Batch score UPI transactions."""
        async for request in request_iterator:
            # Process each request
            yield await self.ScoreUpiTxn(request, context)


if __name__ == "__main__":
    config = OmegaConf.load("configs/upi_local.yaml")
    asyncio.run(serve(config))