## Introduction of Machine Learning - lab6 report by 111062332 朱誼學

#### 1. When predicting values using sine wave data, is there a performance difference between the model that only contains Dense layers and one that includes an RNN layer? Which performs better?  (1%)

<div style="text-align: center;">
    <img src="Q1.png" alt="alt text" width="400">
</div>

- Yes, Despite the fact that the model includes an RNN layer cost more compute time than the one contains only Dense layers, RNN has better performance.


#### 2. Have you tried stacking two consecutive RNN layers in the model? How would you configure the parameters for the second RNN layer if the first RNN layer is defined as RNN(1, 16)? Briefly explain your reasoning.(2%)

``` python
model = Model()
model.add(RNN(input_size, 16))
model.add(RNN(16, 32))
model.add(Dense(32, dense_units))
...
# ValueError: operands could not be broadcast together with shapes (16,16) (16,32) 
```

- If the first RNN is defined as `RNN(1, 16)`, than the second RNN layer must be `RNN(16, 16)`.
  - The first 16 is from the `output_size` of the first RNN layer
  - The second 16 is because we can only implement RNN stacking as long as all the RNN layers (except for the first one) should have same input and output size.
- The reason is that, with the given template, backpropagation of `dL/dh` will have wrong shape.
  - Take the above code as example. Since second layer has 32 as the `output_size`, the backward passed `dL/dh` has shape `(batch_size, 32)`. The shape should be `(batch_size, 16)`, therefore, causing the error shows at the last line of the code section.

#### 3.What would be the effects with the larger size of hidden units in RNN layer? (2%) 

- With the larger size of hidden unit, RNN model will have higher computing time, but also greater performance.
  - Able to learn long-term dependencies.
  - Captures more complex patterns.
  - Larger computational cost
  - More possible for overfitting