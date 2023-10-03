# DataAggregationMI
This repository contains the code and data accompanying our paper '**Not as simple as we thought: a rigorous examination of data aggregation in materials informatics**' (https://chemrxiv.org/engage/chemrxiv/article-details/64d212414a3f7d0c0dced297).

### Installation
1. Clone this repository:
   ```git
   git clone https://github.com/FedeOtto/DataIntegrationMI
   ```
2. Install a new `conda` environment from `daggrmi_env.yml`:
   ```git
   conda env create -f daggrmi_env.yml
   ```
3. Activate the new environment:
   ```git
   conda activate daggrmi
   ```

### Data
MPDS data used to reproduce the experiments can be obtained by using `retrieve_mpds.py` script, given that access to the API is provided. For more info visit https://mpds.io/developer/. Examples of data aggregation can still be reproduced using Materials Project (`mp`) and AFLOW (`aflow`) data assessed in this work.
