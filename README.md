# Quantization of ML Models
## Introduction
AI is part of a major industrial revolution and will profoundly transform society over the coming decades. This technology promises major economic and societal benefits. But this digital transition is taking place at the same time as the advent of environmental issues linked to the climate crisis.

According to a projection by the International Energy Agency, the global data center and AI industries are set to double their electricity consumption by 2026, generating a surplus of 37 billion tonnes of CO2 in the atmosphere. As the use of machine learning explodes in all sectors of society and in all fields, it is vital to find solutions to make these technologies less energy and resource-intensive, and more sustainable.

One such solution is the quantification of ML models, making them less energy-intensive, albeit with slightly less precision or less relevant results.

The aim of this project is to understand the benefits of quantifying Machine Learning models to make them more efficient. The goal of these experiments, then, is to apply the concepts of quantization to different pre-trained models, and to compare the impact of quantization on the efficiency of the models in their respective tasks, as well as on the demand for resources, be they energy, computational or temporal, using different metrics that we'll describe in detail later.

## How quantization works
Quantization is a technique used to reduce the precision of numerical values in a model. Instead of using high-precision data types, such as 32-bit floating-point numbers, quantization represents values using lower-precision data types, such as int8 integers.

## Why using quantization
This process, in addition to significantly reducing memory usage, can speed up model execution while maintaining acceptable accuracy and saving energy resources. In fact, by reducing the precision of weights and activations, each parameter occupies less memory, thus reducing the overall size of the model. Moreover, calculations with integers or half-precision values are faster, enabling more data to be processed in less time. Finally, with less precise calculations, modern processors consume less energy, making the impact of ML models more acceptable.The idea is to try to reduce model power consumption by lowering its precision, while retaining relevance to the task in hand. 

Its use can therefore be very useful in a number of scenarios, for example for experimentation or dimensioning purposes when optimum precision is not required but an ML model needs to be used to test certain things. Another practical application is to enable the use of ML models in embedded systems, which consequently have fewer resources at their disposal. Finally, we can also imagine meeting the needs of real-time applications through the use of quantization, as these require a short response time to be relevant and don't always need to respond to complex tasks.

## Learn more
Read ``Report_ML_quantization.pdf`` for more informations

## Contributors
Noé Caringi
Lucas Girardet
