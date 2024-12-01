import pytest
import os
import torch
import numpy as np
from PIL import Image
from pathlib import Path
import shutil

from model.data import Data


@pytest.fixture
def temp_data_dir(tmp_path):
    """Create a temporary directory structure with dummy image data"""
    # Create class directories
    class1_dir = tmp_path / "class1"
    class2_dir = tmp_path / "class2"
    class1_dir.mkdir()
    class2_dir.mkdir()
    
    # Create dummy images
    for i in range(2):
        img = Image.new('RGB', (100, 100), color='red')
        img.save(class1_dir / f"img{i}.jpg")
        
        img = Image.new('RGB', (100, 100), color='blue')
        img.save(class2_dir / f"img{i}.jpg")
    
    return tmp_path


@pytest.fixture
def data_handler(temp_data_dir):
    """Create a Data instance with the temporary directory"""
    return Data(str(temp_data_dir), train=True)

def test_initialization():
    """Test Data class initialization"""
    input_path = "/dummy/path"
    output_path = "/dummy/output"
    
    # Test with all parameters
    data = Data(input_path, output_path, train=True)
    assert data.input_path == input_path
    assert data.output_path == output_path
    assert data.train == True
    
    # Test with default parameters
    data = Data(input_path)
    assert data.input_path == input_path
    assert data.output_path == None
    assert data.train == True


def test_train_transforms(data_handler):
    """Test training transforms configuration"""
    transforms = data_handler.get_transforms()
    
    # Create a dummy image
    img = Image.new('RGB', (100, 100), color='red')
    
    # Apply transforms
    result = transforms(img)
    
    # Check output properties
    assert isinstance(result, torch.Tensor)
    assert result.shape == (3, 224, 224)  # Channels, Height, Width
    assert result.dtype == torch.float32


def test_eval_transforms():
    """Test evaluation transforms configuration"""
    data = Data("/dummy/path", train=False)
    transforms = data.get_transforms()
    
    # Create a dummy image
    img = Image.new('RGB', (100, 100), color='red')
    
    # Apply transforms
    result = transforms(img)
    
    # Check output properties
    assert isinstance(result, torch.Tensor)
    assert result.shape == (3, 224, 224)
    assert result.dtype == torch.float32


def test_preprocess(data_handler, temp_data_dir):
    """Test image preprocessing"""
    # Get first image path from temp directory
    class1_dir = temp_data_dir / "class1"
    image_path = str(next(class1_dir.iterdir()))
    
    # Create sample image data
    image_data = {
        "image_path": image_path,
        "label": 0
    }
    
    # Process the image
    result = data_handler._preprocess(image_data)
    
    # Check results
    assert "image" in result
    assert "label" in result
    assert isinstance(result["image"], np.ndarray)
    assert result["image"].shape == (3, 224, 224)
    assert result["label"] == 0

def test_create_dataset(data_handler):
    """Test dataset creation and class mapping"""
    dataset, class_to_idx = data_handler.create_dataset()
    
    # Test class mapping
    assert len(class_to_idx) == 2
    assert "class1" in class_to_idx
    assert "class2" in class_to_idx
    assert class_to_idx["class1"] in [0, 1]
    assert class_to_idx["class2"] in [0, 1]
    assert class_to_idx["class1"] != class_to_idx["class2"]
    
    # Test dataset properties using Ray's take() method
    first_batch = dataset.take(1)
    assert len(first_batch) > 0
    first_item = first_batch[0]
    
    assert "image" in first_item
    assert "label" in first_item
    assert isinstance(first_item["image"], np.ndarray)
    assert first_item["image"].shape == (3, 224, 224)
    assert isinstance(first_item["label"], int)
    assert first_item["label"] in [0, 1]



if __name__ == "__main__":
    pytest.main([__file__])