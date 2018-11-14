# Machine Learning Interpretability Catalog
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/BleckwenAI/mli-catalog/master)

Machine learning models are at the core of many recent advances in science and technology affecting our work and our lives. With the growing impact of ML algorithms on society, it is no longer acceptable to trust the model without an answer to the question: why? why did the model make a precise decision? 

This repository provides you with a catalog of new techniques enabling to interpret black box machine learning models. Notebooks detail the use of the following techniques:

* White Box models
  * Linear Models
  * Logistic Regression
  * Generalized Additive Models (GAMs)
* Specific Methods of interpretability
  * Treeinterpreter
  * Tree SHAP - SHapely Additive exPlanations for tree-based models  
* Agnostic Methods of interpretability
  * Local Interpretable Model-agnostic Explanations (LIME)
  * Kernel SHAP - SHapely Additive exPlanations for any classifier/regressor  
   
 
This catalog is under active development and more techniques might be covered in the future. Feel free to contribute with ideas, new approaches or adding your own interpretability method.

### How to play with notebooks

Reading notebooks is easy: Just browser notebooks as Github supports .ipynb rendering. You are able to read all cell outputs (both textual and plots).

However, if you want to execute the notebooks you have two options:

A) (easier and immediate) Use the Binder badge [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/BleckwenAI/mli-catalog/master) to launch an ephemeral Jupyter environment with all dependencies.

B) Have Conda tool installed in your computer.
Clone this repository and then create an environment required by all notebooks:
```
$ conda env create -f environment.yml
```
Then activate the new environment and install a Jupyter Kernel referencing it:
```
$ source activate iml-all
$ python -m ipykernel install --user --name myenv --display-name "IML all (py36)"
```
### License
License: CC BY-NC-SA 4.0

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>.

All notebooks are licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International Public License.

A summary of this license is as follows.

You are free to:

    Share — copy and redistribute the material in any medium or format
    Adapt — remix, transform, and build upon the material

    The licensor cannot revoke these freedoms as long as you follow the license terms.

Under the following terms:

    Attribution — You must give appropriate credit, provide a link to the license, and indicate if changes were made. You may do so in any reasonable manner, but not in any way that suggests the licensor endorses you or your use.

    NonCommercial — You may not use the material for commercial purposes.

    ShareAlike — If you remix, transform, or build upon the material, you must distribute your contributions under the same license as the original.

    No additional restrictions — You may not apply legal terms or technological measures that legally restrict others from doing anything the license permits.


### Maintainers

+ Manar Toumi - creator and contributor
+ Leonardo Noleto - reviewer and contributor