import pytest
import torch
import torch.nn as nn
import numpy as np
from torch.testing import assert_close

from model.architecture import DenseLayer, DenseBlock, TransitionLayer, DenseNet

class TestDenseLayer:
    @pytest.fixture
    def layer(self):
        return DenseLayer(in_channels=64, growth_rate=32)
    
    def test_output_shape(self, layer):
        batch_size = 4
        x = torch.randn(batch_size, 64, 32, 32)
        output = layer(x)
        
        # Output channels should be input channels + growth rate
        expected_shape = (batch_size, 96, 32, 32)
        assert output.shape == expected_shape
        
    def test_concatenation(self, layer):
        x = torch.randn(2, 64, 16, 16)
        output = layer(x)
        
        # First part of output should be identical to input
        assert_close(output[:, :64], x)

class TestDenseBlock:
    @pytest.fixture
    def block(self):
        return DenseBlock(num_layers=4, in_channels=64, growth_rate=32)
    
    def test_output_shape(self, block):
        batch_size = 4
        x = torch.randn(batch_size, 64, 32, 32)
        output = block(x)
        
        # Output channels should be initial channels + (num_layers * growth_rate)
        expected_channels = 64 + (4 * 32)
        expected_shape = (batch_size, expected_channels, 32, 32)
        assert output.shape == expected_shape
        
    def test_layer_connectivity(self, block):
        x = torch.randn(2, 64, 16, 16)
        output = block(x)
        
        # First part of output should be identical to input
        assert_close(output[:, :64], x)

class TestTransitionLayer:
    @pytest.fixture
    def transition(self):
        return TransitionLayer(in_channels=128, out_channels=64)
    
    def test_output_shape(self, transition):
        batch_size = 4
        x = torch.randn(batch_size, 128, 32, 32)
        output = transition(x)
        
        # Spatial dimensions should be halved, and channels should match out_channels
        expected_shape = (batch_size, 64, 16, 16)
        assert output.shape == expected_shape
        
    def test_pooling(self, transition):
        x = torch.ones(2, 128, 32, 32)
        output = transition(x)
        assert output.shape[-1] == x.shape[-1] // 2
        assert output.shape[-2] == x.shape[-2] // 2

class TestDenseNet:
    @pytest.fixture
    def model(self):
        return DenseNet(num_classes=10, growth_rate=12, num_blocks=3, num_layers_per_block=4)
    
    def test_output_shape(self, model):
        batch_size = 4
        x = torch.randn(batch_size, 3, 32, 32)
        output = model(x)
        
        expected_shape = (batch_size, 10)
        assert output.shape == expected_shape
    
    def test_forward_pass(self, model):
        x = torch.randn(2, 3, 32, 32)
        output = model(x)
        
        # Check if output has valid probabilities after softmax
        probs = torch.softmax(output, dim=1)
        assert torch.all(probs >= 0) and torch.all(probs <= 1)
        assert_close(torch.sum(probs, dim=1), torch.ones(2))
    
    @pytest.mark.parametrize("batch_size", [1, 4, 8])
    def test_different_batch_sizes(self, model, batch_size):
        x = torch.randn(batch_size, 3, 32, 32)
        output = model(x)
        assert output.shape == (batch_size, 10)

def test_end_to_end():
    # Test the entire network with random data
    model = DenseNet(num_classes=10)
    x = torch.randn(4, 3, 32, 32)
    
    # Test training mode
    model.train()
    output = model(x)
    assert output.requires_grad
    
    # Test evaluation mode
    model.eval()
    with torch.no_grad():
        output = model(x)
    assert not output.requires_grad

if __name__ == "__main__":
    pytest.main([__file__])