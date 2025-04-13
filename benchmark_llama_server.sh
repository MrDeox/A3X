#!/bin/bash
# Script to benchmark llama-server performance with different ngl values

# Activate virtual environment
source venv/bin/activate

# Define output file for benchmark results
OUTPUT_FILE="benchmark_results.txt"

# Define the model path
MODEL_PATH="/home/arthur/projects/A3X/models/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf"

# Define different ngl values to test
NGL_VALUES=(0 15 25 33 35)

# Define a simple test prompt
TEST_PROMPT="Hello, how are you?"

# Clear previous benchmark results
echo "Benchmark Results for llama-server with Meta-Llama-3-8B-Instruct.Q4_K_M.gguf" > $OUTPUT_FILE
echo "Date: $(date)" >> $OUTPUT_FILE
echo "Hardware: AMD Radeon RX 6400 (4GB VRAM), Intel i5 11th Gen" >> $OUTPUT_FILE
echo "----------------------------------------" >> $OUTPUT_FILE

# Function to measure response time for a given ngl value
measure_response_time() {
    local ngl=$1
    echo "Testing with ngl=$ngl..."
    echo "----------------------------------------" >> $OUTPUT_FILE
    echo "Test with ngl=$ngl" >> $OUTPUT_FILE
    echo "----------------------------------------" >> $OUTPUT_FILE

    # Start the llama-server with the specified ngl value
    /home/arthur/projects/A3X/llama.cpp/build/bin/llama-server -m $MODEL_PATH -ngl $ngl &
    SERVER_PID=$!

    # Wait a bit for the server to start
    sleep 10

    # Measure start time
    START_TIME=$(date +%s.%N)

    # Send a request to the server
    RESPONSE=$(curl -s -X POST http://127.0.0.1:8080/completion -H "Content-Type: application/json" -d "{\"prompt\": \"$TEST_PROMPT\", \"n_predict\": 50}")

    # Measure end time
    END_TIME=$(date +%s.%N)

    # Calculate duration
    DURATION=$(echo "$END_TIME - $START_TIME" | bc)

    # Extract response content length
    CONTENT_LENGTH=$(echo $RESPONSE | grep -o '"content": ".*"' | wc -c)

    # Log results
    echo "Response Time: $DURATION seconds" >> $OUTPUT_FILE
    echo "Content Length: $CONTENT_LENGTH characters" >> $OUTPUT_FILE
    echo "Response: $RESPONSE" >> $OUTPUT_FILE
    echo "" >> $OUTPUT_FILE

    # Stop the server
    kill $SERVER_PID
    wait $SERVER_PID 2>/dev/null
}

# Run benchmark for each ngl value
for ngl in "${NGL_VALUES[@]}"; do
    measure_response_time $ngl
    sleep 5  # Give some time between tests to avoid port conflicts
done

# Deactivate virtual environment
deactivate

# Print completion message
echo "Benchmark completed. Results are saved in $OUTPUT_FILE" 