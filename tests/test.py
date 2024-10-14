from guv_calcs import *

room = Room(x=6, y=4,z=2.7, units='meters')
# df=get_disinfection_table(1,280,room)
# print(df)

lamp = Lamp("Lamp1", 
            filename="src/guv_calcs/data/lamp_data/uvpro222_b1.ies",
            spectra_source="src/guv_calcs/data/lamp_data/uvpro222_b1.csv",
            ).move(x=room.x/2,y=room.y/2,z=2.6)
height = 1.9
skin_limits = CalcPlane("SkinLimits", height = height,x1=0,x2=room.x,y1=0,y2=room.y, vert=False, horiz=False, fov80=False, dose=True)
eye_limits = CalcPlane("EyeLimits", height = height,x1=0,x2=room.x,y1=0,y2=room.y, vert=True, horiz=False, fov80=True, dose=True)
fluence = CalcVol(
        zone_id="WholeRoomFluence",
        x1=0,
        x2=room.x,
        y1=0,
        z1=0,
        z2=room.z,
    )
room.add_calc_zone(fluence)
room.add_calc_zone(skin_limits)
room.add_calc_zone(eye_limits)
room.add_lamp(lamp)
room.calculate()

print(get_disinfection_table(fluence.values.mean()))
standard = get_standards()[1]
print(get_tlv(222,standard))

spectra_source="src/guv_calcs/data/lamp_data/uvpro222_b1.csv"
spec = Spectrum.from_file(spectra_source)
print(spec.get_tlv(standard))
print(spec.get_max_irradiance(standard))
print(spec.get_seconds_to_tlv(15, standard)/3600)

room.add_lamp(lamp).add_standard_zones().calculate()


room.save('tests/test.guv')
newroom = Room.load('tests/test.guv')


skin_limits.export()
fluence.export()

room.export_zip("tests/test.zip",
    include_plots=True,
    include_lamp_files=True,
    include_lamp_plots=True,)

# # import matplotlib.pyplot as plt
# # # fig=skin_limits.plot_plane()
# # # plt.show()
# # lamp.plot_spectra()
# # plt.show()
# # # fig=room.plotly()
# # fig.show()

# # room.to_json('tests/test.json')
# # newroom = Room.from_json('tests/test.json')
# # newroom.calculate()

# test data retrieval functions
