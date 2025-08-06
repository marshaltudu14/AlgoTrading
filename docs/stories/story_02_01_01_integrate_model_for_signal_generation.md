---

### **Story 2.1.1: Integrate Model for Signal Generation**

**Status:** `Ready for dev`

**Story:**
As a developer, I need to integrate the pre-trained PyTorch model (`universal_final_model.pth`) into the `LiveTradingService` to generate trading signals from the processed market data.

**Acceptance Criteria:**
1.  The `LiveTradingService` successfully loads the `universal_final_model.pth` file from the `models/` directory upon initialization.
2.  At each time-based interval, the `LiveTradingService` takes the newly fetched and processed historical data (as a pandas DataFrame).
3.  The data is converted into a PyTorch tensor with the correct shape and data type expected by the model.
4.  The tensor is fed into the model to get a prediction.
5.  The model's output (a tensor) is correctly parsed to extract the trading signal (an integer representing buy, sell, hold, etc.) and the predicted quantity.
6.  The extracted signal and quantity are stored in variables for use in subsequent steps.
7.  The system logs an error if the model file cannot be loaded or if the model prediction fails.

**Tasks / Subtasks:**
-   `[ ]` **Backend:** In `src/trading/live_trading_service.py`, add the necessary imports for `torch` and the model class from `src/models/`.
-   `[ ]` **Backend:** In the `__init__` method of `LiveTradingService`, add the logic to load the model from `models/universal_final_model.pth` and set it to evaluation mode (`model.eval()`).
-   `[ ]` **Backend:** In the main trading loop of the service, after data processing, implement the logic to convert the pandas DataFrame to a PyTorch tensor.
-   `[ ]` **Backend:** Ensure the tensor is properly formatted (e.g., normalized, correct dimensions) as required by the model's input layer.
-   `[ ]` **Backend:** Call the model with the input tensor: `model(input_tensor)`.
-   `[ ]` **Backend:** Implement the logic to decode the output tensor into a human-readable signal and a numerical quantity.
-   `[ ]` **Backend:** Add logging to record the generated signal and quantity at each interval.
-   `[ ]` **Testing:** In `tests/test_trading/`, create `test_live_trading_service_model_integration.py`.
-   `[ ]` **Testing:** Write a unit test that initializes the `LiveTradingService` and verifies that the model is loaded correctly.
-   `[ ]` **Testing:** Write a test that provides a sample DataFrame to the service and asserts that the model is called and that a valid signal and quantity are produced. (This will require mocking the model's output).

---