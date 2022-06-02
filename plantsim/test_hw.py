import pandas as pd

from plantsim.plantsim import Plantsim
from plantsim.pandas_table import PandasTable
import time
import pandas

model = 'F:\Tjark\Dokumente\FH Bielefeld\Master\SoSe2022\Diskrete Simulation und Reinforcement Learning\PlantSim Dateien\Modell1.spp'
plantsim = Plantsim(version='16.1',license_type='Educational',path_context='.Modelle.Modell',model=model, socket = None, visible = False)

#Starting simulation and letting it work
plantsim.start_simulation()
time.sleep(3)

#Getting the result table from PlantSim
table = plantsim.get_object("Results")
print(table)

#Manipulating the data
dict = table.rows_coldict
dict[0]["SimTime"]= 10
print(dict)

#Getting the result table from PlantSim with Pandas
table2 = PandasTable(plantsim,"Results") #, 'F:\Tjark\Dokumente\FH Bielefeld\Master\SoSe2022\Diskrete Simulation und Reinforcement Learning\PlantSimPython'
print(table2.table)
df = table2.table

#Manipulating the data
df.at[0,'SimTime'] = 100
print(df)

plantsim.quit()