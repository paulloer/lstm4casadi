# **lstm4casadi**
A custom LSTM implementation designed for use with CasADi. While the [l4casadi library](https://github.com/Tim-Salzmann/l4casadi) may be preferable for general purposes, this repository provides an alternative, which makes LSTM's accessible in CasADi.

---

## **Motivation**  
As the recurrent neural network (RNN) architechture long short-term memory (LSTM) is not compatible with `l4casadi`, I instead implemented it on my own.

---

## **Implementation**  

This implementation adheres closely to the LSTM description in the [PyTorch documentation](https://pytorch.org/docs/stable/nn.html#torch.nn.LSTM). To ensure correctness and reliability, the implementation includes two variants:

1. **CasADi-Based LSTM:**  
   This version uses CasADi symbolic expressions and `Opti` module variables to enable integration with CasADi's optimization framework.

2. **PyTorch-Based LSTM:**  
   This version replicates the LSTM functionality in PyTorch, allowing the model's behavior and performance to be verified against a well-established framework.

### **Features**
- **Multi-Layer Support**: The LSTM supports multiple layers for capturing complex sequential dependencies.  
- **Final Fully-Connected Layer**: Enables a straightforward mapping of outputs to desired dimensions.  
- **Simplified Configuration**: For simplicity and alignment with the project requirements, the following parameters are fixed:  
  - `bidirectional == False`  
  - `dropout == 0.0`  
  - `proj_size == 0`  

---

## **Getting Started**  
### **Dependencies**  
- [CasADi](https://web.casadi.org)  
- [PyTorch](https://pytorch.org)  

### **Installation**  
Clone this repository:  
```bash
git clone https://github.com/yourusername/lstm4casadi.git
cd lstm4casadi
```

### **Acknowledgments**

[l4casadi ](https://github.com/Tim-Salzmann/l4casadi) for inspiring this project.
PyTorch documentation for serving as the basis for the LSTM implementation.

### **License**
 
The MIT License (MIT)

Copyright (c) 2024 Paul Loer

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
