from plantsim.plantsim import Plantsim

model = 'F:\Tjark\Dokumente\FH Bielefeld\Master\SoSe2022\Diskrete Simulation und Reinforcement Learning\GitHub Files\DiskreteSim_ReinfLearn\plantsim_deepql\MiniFlow_BE_based_MAS.spp'

plantsim = Plantsim(version='16.1',license_type='Educational',path_context='.Modelle.Modell',model=model, socket = None, visible = False)

plantsim.start_simulation()

# Funktioniert
print(plantsim.get_current_state())

plantsim.quit()

