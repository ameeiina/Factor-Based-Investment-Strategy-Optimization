# Factor-Based-Investment-Strategy-Optimization
This repository is a collection of black bod optimization algorithms that stemmed from my
[Master Thesis](http://hdl.handle.net/10362/19974). The original code was completely refactored and is
now made available through this project. 

It allows you to easily build and train neural networks in Java; you can also extend the core functionality for particular use-cases.WIP

It includes a variety of features and visualizations to help you create powerful and accurate models for your data.WIP

### Models
Self-Organizing Maps, Gaussian Processes and Genetic Algorithms Applied to Factor-Based Investment Strategy Optimization.


This project implements three types of **neural network models and variants**:

- **`SelfOrganizingMap`** (SOM): a type of neural network that projects high-dimensional data onto a 2d map, maintaining input-space topological relationships. Useful for exploratory cluster analysis and, possibly, classification.
  - `BasicSOM` - *Classical* and *Batch* algorithms are available;
  - For *streaming data*, i.e., variants that can learn incrementally over time, estimating its learning parameters on-the-fly:
    - `UbiSOM` - The Ubiquitous Self-Organizing Map (*a contribution from my thesis*);
    - `PLSOM` - The Parameterless Self-Organizing Map;
    - `DSOM` - The Dynamic Self-Organizing Map.

  > All SOM models allow setting the lattice type (hexagonal or rectangular) and the metric distance to use (euclidean or manhattan); but you
  > can easily create your own lattices and metric distances. 

- **GP** : gp
- **Genetic algorithm**: kbcs

### Data and preprocessing

- Data is imported through the `Dataset` class. Data set files must have a *yaml* header describing it - see examples in the `datasets` folder;
- There are two implementations of `DatasetNormalization`, namely `MinMaxNormalization` and `MeanNormalization`. You can add others;
- The class `DatasetTrainSplit`, as the name suggests, allows you to easily split your data for training and testing;
- The library comes with an implementation of `PCA` data projection.

### Visualizations

Contains different types of visualizations for all available models and a simple plotting class.

#### Self-Organizing Maps

Add photo

- U-matrix
- Component planes
- Hit-Map
- Spatial visualization of input data and lattice
- Clustering of prototypes
- ...

#### GP

Add photo

- Spatial visualization of input data and generated micro-categories

#### MOGA

Add photo

- Network architecture, weights and bias

## Documentation

### Installation

Binaries and dependency information for Maven, Gradle and others can be found at WIP
Ask for Qrumble or write about its dependency!

Example for Maven:

```xml
<dependency>
    <groupId>com.brunomnsilva</groupId>
    <artifactId>neuralnetworks</artifactId>
    <version>x.y.z</version>
</dependency>
```
WIP

You need Java 9 or later.
You need Python 3 or later 


Some *transient* dependencies are used, namely:

- [snakeyaml](https://mvnrepository.com/artifact/org.yaml/snakeyaml)
- [colt](https://mvnrepository.com/artifact/colt/colt)
- [commons-math3](https://mvnrepository.com/artifact/org.apache.commons/commons-math3)
- [jcommon](https://mvnrepository.com/artifact/org.jfree/jcommon)
- [jfreechart](https://mvnrepository.com/artifact/org.jfree/jfreechart)
- [jfreesvg](https://mvnrepository.com/artifact/org.jfree/jfreesvg)
- [JMathPlot](https://mvnrepository.com/artifact/com.github.yannrichet/JMathPlot)

  WIP

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. 
All derivative work should include this license.

## Authors

Original author: **Amina Enkhbaatar** 


---

I hope you find this repository useful and look forward to seeing the projects you create with it!
