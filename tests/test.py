from guv_calcs import *
import numpy as np

room = Room(x=6, y=4,z=2.7, units='meters')
# df=get_disinfection_table(1,280,room)
# print(df)

# lamp = Lamp("Lamp1", 
            # filename="src/guv_calcs/data/lamp_data/uvpro222_b1.ies",
            # spectra_source="src/guv_calcs/data/lamp_data/uvpro222_b1.csv",
            # # units='meters',
            # # length=0.1,
            # # width=0.1,
            # source_density=1,
            # ).move(x=room.x/2,y=room.y/2,z=2.6)
            
# print(lamp.grid_points,lamp.photometric_distance)
# height = 1.9
# skin_limits = CalcPlane("SkinLimits", height = height,x1=0,x2=room.x,y1=0,y2=room.y, vert=False, horiz=False, fov80=False, dose=True)
# eye_limits = CalcPlane("EyeLimits", height = height,x1=0,x2=room.x,y1=0,y2=room.y, vert=True, horiz=False, fov80=True, dose=True)
# fluence = CalcVol(
        # zone_id="WholeRoomFluence",
        # x1=0,
        # x2=room.x,
        # y1=0,
        # z1=0,
        # z2=room.z,
    # )
# room.add_calc_zone(fluence)
# room.add_calc_zone(skin_limits)
# room.add_calc_zone(eye_limits)

rel_map = np.array([
	0.679274542,1.004584682,0.804004854,0.758158037,0.532632147,
	0.748381877,1.439455232,1.303600324,1.109088457,0.666127292,
	0.768608414,1.520361381,1.498786408,1.277642934,0.679948759,
	0.824905609,1.554072276,1.380461165,1.179881338,0.645563646,
	0.765237325,1.183252427,1.061893204,1.007955771,0.606121899,
])

import matplotlib.pyplot as plt
for val in [1,2,3]:
    lamp = Lamp("Lamp1", 
                filename="src/guv_calcs/data/lamp_data/uvpro222_b1.ies",
                spectra_source="src/guv_calcs/data/lamp_data/uvpro222_b1.csv",
                angle=270,
                source_density=val,
                relative_map=rel_map if val==3 else None
                ).move(1,1,1)
    zone = CalcPlane("test", 
            height = 0.95,
            x1=0.9,x2=1.1,
            y1=0.9,y2=1.1,
            x_spacing=0.01,y_spacing=0.01)

    room.add_lamp(lamp).add_calc_zone(zone)
    room.calculate()
    
    fig,ax=zone.plot_plane()
    plt.show()

# # test data retrieval functions
# print(get_disinfection_table(fluence.values.mean()))
# standard = get_standards()[1]
# print(get_tlv(222,standard))

# spectra_source="src/guv_calcs/data/lamp_data/uvpro222_b1.csv"
# spec = Spectrum.from_file(spectra_source)
# print(spec.get_tlv(standard))
# print(spec.get_max_irradiance(standard))
# print(spec.get_seconds_to_tlv(15, standard)/3600)

# room.add_lamp(lamp).add_standard_zones().calculate()


# room.save('tests/test.guv')
# newroom = Room.load('tests/test.guv')


# skin_limits.export()
# fluence.export()

# room.export_zip("tests/test.zip",
    # include_plots=True,
    # include_lamp_files=True,
    # include_lamp_plots=True,)
    
    


# # import matplotlib.pyplot as plt
# # fig=skin_limits.plot_plane()
# # plt.show()
# lamp.plot_spectra()
# plt.show()
# # fig=room.plotly()
# fig.show()

# room.to_json('tests/test.json')
# newroom = Room.from_json('tests/test.json')
# newroom.calculate()

