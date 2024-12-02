import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile
import ray
from torch.nn.parallel.distributed import DistributedDataParallel

from architecture import DenseNet
from train import one_train_step, one_eval_step, train_func_per_worker
from data import Data

@pytest.fixture
def model():
    return DenseNet(in_channels=3, num_classes=10)

@pytest.fixture
def sample_data():
    images = torch.randn(4, 3, 32, 32)
    labels = torch.randint(0, 10, (4,))
    return {"images": images, "labels": labels}

def test_densenet_forward(model):
    x = torch.randn(2, 3, 32, 32)
    out = model(x)
    assert out.shape == (2, 10)

def test_densenet_save(model):
    with tempfile.TemporaryDirectory() as tmp_dir:
        model.save(tmp_dir)
        assert Path(tmp_dir, "model.pt").exists()

def test_densenet_save_ddp(model):
    if torch.cuda.is_available():
        model = model.cuda()
        model = DistributedDataParallel(model)
        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save(tmp_dir)
            assert Path(tmp_dir, "model.pt").exists()

def test_one_train_step(model, sample_data):
    class MockDataset:
        def iter_torch_batches(self, batch_size):
            yield sample_data
    
    optimizer = torch.optim.AdamW(model.parameters())
    loss_fn = torch.nn.CrossEntropyLoss()
    
    loss = one_train_step(
        ds=MockDataset(),
        batch_size=2,
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer
    )
    assert isinstance(loss, float)

def test_one_eval_step(model, sample_data):
    class MockDataset:
        def iter_torch_batches(self, batch_size):
            yield sample_data
    
    loss_fn = torch.nn.CrossEntropyLoss()
    
    loss, y_true, y_pred = one_eval_step(
        ds=MockDataset(),
        batch_size=2,
        model=model,
        loss_fn=loss_fn
    )
    assert isinstance(loss, float)
    assert isinstance(y_true, np.ndarray)
    assert isinstance(y_pred, np.ndarray)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_train_func_per_worker():
    if ray.is_initialized():
        ray.shutdown()
    ray.init()
    
    config = {
        "num_epochs": 1,
        "lr": 0.01,
        "lr_factor": 0.1,
        "lr_patience": 3,
        "batch_size": 2,
        "num_classes": 10
    }
    
    try:
        train_func_per_worker(config)
    except Exception as e:
        pytest.fail(f"Training function failed: {str(e)}")
    finally:
        ray.shutdown()

def test_data_loading():
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create mock data structure
        data_dir = Path(tmp_dir)
        (data_dir / "class1").mkdir()
        (data_dir / "class2").mkdir()
        
        # Create dummy image files
        (data_dir / "class1" / "img1.jpg").touch()
        (data_dir / "class2" / "img1.jpg").touch()
        
        data = Data(tmp_dir)
        dataset, class_to_idx = data.create_dataset()
        
        assert len(class_to_idx) == 2
        assert dataset is not None