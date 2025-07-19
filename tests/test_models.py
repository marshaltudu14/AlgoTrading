import unittest
import torch
from src.models.lstm_model import LSTMModel

class TestLSTMModel(unittest.TestCase):

    def test_model_output_shape(self):
        input_dim = 10
        hidden_dim = 20
        output_dim = 5
        num_layers = 1
        batch_size = 32
        sequence_length = 50

        model = LSTMModel(input_dim, hidden_dim, output_dim, num_layers)

        # Test with 3D input (batch_size, sequence_length, input_dim)
        input_tensor_3d = torch.randn(batch_size, sequence_length, input_dim)
        output_3d = model(input_tensor_3d)
        self.assertEqual(output_3d.shape, (batch_size, output_dim))

        # Test with 2D input (batch_size, input_dim) - should be unsqueezed to (batch_size, 1, input_dim)
        input_tensor_2d = torch.randn(batch_size, input_dim)
        output_2d = model(input_tensor_2d)
        self.assertEqual(output_2d.shape, (batch_size, output_dim))

    def test_model_instantiation(self):
        try:
            model = LSTMModel(input_dim=10, hidden_dim=20, output_dim=5)
            self.assertIsInstance(model, LSTMModel)
        except Exception as e:
            self.fail(f"Model instantiation failed: {e}")

if __name__ == '__main__':
    unittest.main()
