# energy-influence-model-structure-uncertainty


## GP4P - Gaussian Process Regression for Performance Prediction

Performance optimization for configurable software systems is not only an im-
portant aspect of software engineering, it restricts how the software is being
used on the base of performance evaluations of its configuration options. The
lack of domain knowledge in a black-box scenario induces uncertainty in the de-
cision making process, which is why there is a risk that a certain configuration
exceeds the user’s budget unexpectedly.

To estimate the model structure uncertainty in performance influence mod-
els to identify synergistic effects among selected options, this thesis provides
the methodology of Gaussian Process Regression, which is the foundation of
the Bayesian Optimization Algorithm. The key contribution is the implemen-
tation of a variety of different Gaussian process models with shift invariant
as well as additive structured kernels that are estimated using exact infer-
ence of the marginal log likelihood or the state of the art NUTS sampler for
approximate posterior evaluations.

With the presented methods of the Bayesian Approximate Kernel Regres-
sion and the distance measure groupRATE, the provided application can iden-
tify interactions among configuration options to choose from in a variable se-
lection procedure as a set of constraints to state a dynamic program for finding
a best subset selection

## Installation
- prerequisites
  - Python 3.9.6 or higher
  - pip or
  - conda
In the root directory of the project, run the following command to install the required packages:
```bash
pip install -r requirements.txt
```
or 
```bash
conda install --file requirements.txt
```
or using the setup.py
```bash
python setup.py install
```

## Usage
This project follows the Clean Architecture design principles. The `adapters` directory contains third party module implementations.
The `domain` directory contains the domain logic of the application. Most importantly, `domain/env.py` contains the environment variables that are used throughout the application. See the description there how to adjust them. The `application` directory contains the use cases of the application. To reproduce the data in the thesis, run the following command:
```bash
python application/thesis_pipeline.py
```
or use `application/gpytorch_pipeline.py` to train single models using exact inference. `application/fully_bayesian_pipeline.py` contains the implementation of the fully Bayesian model using the NUTS sampler.

For inference, the `application/model_inference.py`provide insights in the model, especially the BAKR methodology and the groupRATE distance measure. The `application/eval_training.py` provides the evaluation of the models using different score metrics.

## License
The project is published under the GNU General Public License v3.0. If you need to use the project under a different license, please contact the author.

## Project status
The project is still under development. The current version is a prototype that is used to evaluate the performance of the Gaussian Process models, while the methods for inference and evaluation focusses on MCMC models, but work partially also for exact inference models.