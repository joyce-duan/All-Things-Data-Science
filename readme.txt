to set up configuration:

cd All-Things-Data-Science
vi allds.config


cp allds.config webapp/.
cp allds.config code/db/.
cp allds.config code/model/.
cp allds.config code/preprocess/.
cp allds.config code/recommender/.
cp allds.config code/scraper/.
cp allds.config code/eda/.

   

from configobj import ConfigObj
config=ConfigObj('allds.config')
import sys
sys.path.append(allds_home  + 'code/model')



folder structure:
data/  data files (models) used by both web app and analysis code
code/   analysis and modeling
webapp/