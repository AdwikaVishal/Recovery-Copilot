#!/usr/bin/env bash
set -euo pipefail

PORT=${2:-8000}
DIR="$(cd "$(dirname "$0")/backend" && pwd)"

usage() {
    echo "Usage: $0 [api|pipeline|all|live|generate] [port]"
    echo ""
    echo "Commands:"
    echo "  api        Start API server only (preserves database)"
    echo "  pipeline   Run BATCH BENCHMARK pipeline only (preserves database)"
    echo "  all        Run benchmark then start API"
    echo "  live       Start API in LIVE RECOVERY MODE (independent of benchmark)"
    echo "  generate   Generate sample data only"
    echo ""
    echo "Options:"
    echo "  port       Server port (default: 8000)"
    echo ""
    echo "Real-time (live recovery) simulation:"
    echo "  Live events POST through the secure webhook ingress (the simulator attaches"
    echo "  the demo HMAC signature automatically):"
    echo "    python3 -m tools.simulate_realtime --interval 3 --count 10"
    echo ""
    echo "Mode:"
    echo "  BENCHMARK MODE  (pipeline)  — deterministic 100-event batch + 20/20 scenarios"
    echo "  LIVE RECOVERY    (api/live) — webhook ingress, closed-loop, SSE stages"
    echo ""
    echo "Environment:"
    echo "  WEBHOOK_MODE=demo|production   (default: demo)"
    echo "  WEBHOOK_SECRET=<shared secret> (required for production)"
    exit 0
}

cmd="${1:-all}"

if [[ "$cmd" == "--help" || "$cmd" == "-h" ]]; then
    usage
fi

if [[ "$cmd" =~ ^[0-9]+$ ]]; then
    PORT="$cmd"
    cmd="all"
fi

cd "$DIR"

echo "Recovery Copilot v4.0 — Live Recovery"
echo "Working directory: $DIR"
echo ""

run_pipeline() {
    echo "Running batch pipeline..."

    PYTHONPATH=. python3 -c "
import asyncio
from data.generator import generate_batch, save_batch
from engine.pipeline import process_batch, load_batch

async def run():
    try:
        events = load_batch()
    except FileNotFoundError:
        print('No sample data found. Generating...')
        events = generate_batch(100)
        save_batch(events)

    result = await process_batch(events)

    print(f'Batch processed: {result.recovered}/{result.total_records} recovered')
    print(f'Amount: ₹{result.recovered_amount // 100:,} (baseline: ₹{result.baseline_amount // 100:,})')
    print(f'Blocked: {result.blocked_by_policy}, Human Review: {result.human_review}, Errors: {result.errors}')

asyncio.run(run())
"

    echo ""
}

generate_data() {
    echo "Generating sample data..."

    PYTHONPATH=. python3 -m data.generator

    echo ""
}

case "$cmd" in
    generate)
        generate_data
        ;;

    pipeline)
        generate_data
        run_pipeline
        ;;

    api|live)
        if [[ "$cmd" == "live" ]]; then
            echo "⚠  LIVE RECOVERY MODE — webhook ingress + closed-loop + SSE stages"
        else
            echo "API MODE — preserves benchmark; live recovery available via webhooks"
        fi
        echo "Starting API server on http://localhost:$PORT"
        echo "Dashboard: http://localhost:$PORT/"
        echo "API Docs:  http://localhost:$PORT/docs"
        echo "SSE Stream: http://localhost:$PORT/api/events/stream"
        echo "Webhook:   POST http://localhost:$PORT/api/webhooks/payment  (signed)"
        echo "Live metrics: http://localhost:$PORT/api/live/metrics"
        echo ""

        export PYTHONPATH=.
        exec python3 -m uvicorn app.main:app \
            --host 0.0.0.0 \
            --port "$PORT" \
            --reload
        ;;

    all)
        generate_data
        run_pipeline

        echo "Starting API server on http://localhost:$PORT"
        echo "Dashboard: http://localhost:$PORT/"
        echo "API Docs:  http://localhost:$PORT/docs"
        echo "SSE Stream: http://localhost:$PORT/api/events/stream"
        echo "Webhook:   POST http://localhost:$PORT/api/webhooks/payment  (signed)"
        echo "Live metrics: http://localhost:$PORT/api/live/metrics"
        echo ""

        export PYTHONPATH=.
        exec python3 -m uvicorn app.main:app \
            --host 0.0.0.0 \
            --port "$PORT" \
            --reload
        ;;

    *)
        echo "Unknown command: $cmd"
        usage
        ;;
esac
