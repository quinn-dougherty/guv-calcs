from pathlib import Path
from guv_calcs._plot import plot_tlvs
from guv_calcs.room import Room
from guv_calcs.lamp import Lamp
from guv_calcs.calc_zone import CalcVol, CalcPlane

root = Path("./ies_files/")

# initialize room size and units
room = Room(dimensions=(5, 5, 2.5), units='meters')
# add and place lamps
room.add_lamp(Lamp(root / "B1 module.ies").move(1.66,2.5,2.5))
room.add_lamp(Lamp(root / "B1 module.ies").move(3.33,2.5,2.5))

# add calculation zone
room.add_calc_zone(CalcVol("Room Fluence", dimensions=room.dimensions))
# calculate!
room.calculate()
# find average fluence
avg_fluence = room.calc_zones["Room Fluence"].values.mean()
print('Average fluence:', round(avg_fluence,3),'uW/cm²')

# aim the lamps we set up earlier
room.lamps[0].aim(0,0,0)
room.lamps[1].aim(5,5,0)
# calculate!
room.calculate()
# find average fluence
avg_fluence = room.calc_zones["Room Fluence"].values.mean()
print('Average fluence:', round(avg_fluence,3),'uW/cm²')

# room.plot(use_plotly=True)
room.plot()

# plot TLVs
height = 1.9
room.add_calc_zone(CalcPlane("Skin", height=height, dimensions=[room.x,room.y], vert=False, horiz=True, fov80=False))
room.add_calc_zone(CalcPlane("Eye", height=height, dimensions=[room.x,room.y], vert=True, horiz=False, fov80=True))

room.calculate()
eye_values = room.calc_zones["Eye"].values*3.6*8
skin_values = room.calc_zones["Skin"].values*3.6*8

plot_tlvs(skin_values, eye_values, room.x, room.y, height, room.units)