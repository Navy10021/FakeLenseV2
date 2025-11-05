# FakeLenseV2 Notebooks

This directory contains Jupyter notebooks for learning and experimenting with FakeLenseV2.

## 📚 Available Notebooks

### Examples Directory

1. **[01_quick_start.ipynb](examples/01_quick_start.ipynb)**
   - Introduction to FakeLenseV2
   - Loading models and making predictions
   - Single and batch inference
   - Understanding results
   - **Difficulty**: Beginner
   - **Time**: 15-20 minutes

2. **[02_training_guide.ipynb](examples/02_training_guide.ipynb)**
   - Training a model from scratch
   - Data preparation
   - Configuration setup
   - Monitoring training progress
   - **Difficulty**: Intermediate
   - **Time**: 30-45 minutes

## 🚀 Getting Started

### Prerequisites

```bash
# Install Jupyter
pip install jupyter notebook ipywidgets

# Or use JupyterLab
pip install jupyterlab
```

### Running Notebooks

```bash
# From project root
cd notebook
jupyter notebook

# Or JupyterLab
jupyter lab
```

Then open the desired notebook from the browser interface.

### Alternative: VS Code

If you use VS Code:
1. Install the Jupyter extension
2. Open the `.ipynb` file
3. Select Python kernel
4. Run cells

## 📖 Notebook Contents

### 01 - Quick Start Tutorial

```
1. Setup and Installation
2. Load the Model
3. Single Prediction
4. Batch Predictions
5. Understanding Source Reliability
6. Analyzing Classification Patterns
7. Next Steps
```

### 02 - Training Guide

```
1. Setup
2. Prepare Training Data
3. Load Training Data
4. Configure Training Parameters
5. Initialize Components
6. Create Trainer and Start Training
7. Visualize Training Progress
8. Test the Trained Model
9. Save Configuration
10. Tips for Better Training
```

## 💡 Tips

### For Beginners

- Start with `01_quick_start.ipynb`
- Run cells sequentially (Shift + Enter)
- Read the markdown cells for context
- Experiment by modifying the code

### For Advanced Users

- Use `02_training_guide.ipynb` to train custom models
- Modify hyperparameters to experiment
- Create your own notebooks for specific use cases

## 🔧 Troubleshooting

### Kernel Issues

```bash
# Install kernel
python -m ipykernel install --user --name=fakelense

# Select kernel in Jupyter: Kernel > Change Kernel > fakelense
```

### Import Errors

```bash
# Make sure you're in the notebook directory
cd /path/to/FakeLenseV2/notebook

# And the package is installed
pip install -e ..
```

### GPU Not Available

- Check: `torch.cuda.is_available()`
- If False, you're using CPU (slower but still works)
- For GPU: Install CUDA-enabled PyTorch

## 📊 Example Outputs

### Quick Start Notebook

- Predictions on sample articles
- Source reliability comparisons
- Effect of social reactions

### Training Guide Notebook

- Training progress logs
- Loss curves
- Model checkpoints
- Performance metrics

## 🎯 Learning Path

**Recommended Order:**

1. **Week 1**: Complete `01_quick_start.ipynb`
   - Understand basic inference
   - Experiment with different inputs
   - Learn about source reliability

2. **Week 2**: Complete `02_training_guide.ipynb`
   - Prepare custom dataset
   - Train your first model
   - Evaluate results

3. **Week 3**: Advanced experimentation
   - Fine-tune hyperparameters
   - Try different model architectures
   - Deploy your model

## 📚 Additional Resources

- [Main README](../README.md)
- [Usage Guide](../README_USAGE.md)
- [Contributing Guide](../CONTRIBUTING.md)
- [API Documentation](http://localhost:8000/docs)

## 🤝 Contributing

Have ideas for new notebooks?

1. Create your notebook in `examples/`
2. Follow the naming convention: `XX_description.ipynb`
3. Add it to this README
4. Submit a pull request

**Notebook Template Structure:**
```markdown
# Title
Introduction and objectives

## Section 1
Content

## Section 2
Content

## Next Steps
Links to related resources
```

## 📞 Support

- Issues: [GitHub Issues](https://github.com/Navy10021/FakeLenseV2/issues)
- Discussions: [GitHub Discussions](https://github.com/Navy10021/FakeLenseV2/discussions)
- Email: iyunseob4@gmail.com

---

Happy Learning! 🚀
