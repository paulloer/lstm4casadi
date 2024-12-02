# **LSTM4CasADi**
A custom LSTM implementation designed for use with CasADi. While the [l4casadi library](https://github.com/Tim-Salzmann/l4casadi) may be preferable for general purposes, this repository provides an alternative solution, specifically tailored for cases where standard LSTM architectures are incompatible.

---

## **Motivation**  
The long short-term memory (LSTM) recurrent neural network architecture is not natively supported by `l4casadi`. To address this limitation, I developed a custom implementation, leveraging the flexibility of CasADi's symbolic computation capabilities.

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

## **Why Use This Implementation?**  
While other solutions, like `l4casadi`, offer general-purpose compatibility, this implementation is ideal for scenarios requiring:  
- Symbolic computation in CasADi with LSTMs.  
- Close alignment with PyTorch-style LSTM functionality for easy validation and testing.  

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
