from guv_calcs import *

room = Room(x=6,y=4,z=2.7, units='meters')

lamp = Lamp("Lamp1", 
            filename="./tests/uvpro222_b1.ies",
            spectra_source="./tests/uvpro222_b1_spectrum.csv",
            spectral_weight_source="./data/UV Spectral Weighting Curves.csv").move(x=room.x/2,y=room.y/2,z=2.6)
height = 1.9
skin_limits = CalcPlane("SkinLimits", height = height,x1=0,x2=room.x,y1=0,y2=room.y, vert=False, horiz=False, fov80=False, dose=True)
eye_limits = CalcPlane("EyeLimits", height = height,x1=0,x2=room.x,y1=0,y2=room.y, vert=True, horiz=False, fov80=True, dose=True)
fluence = CalcVol(
        zone_id="WholeRoomFluence",
        name="Whole Room Fluence",
        x1=0,
        x2=room.x,
        y1=0,
        y2=room.y,
        z1=0,
        z2=room.z,
    )
room.add_calc_zone(fluence)
room.add_calc_zone(skin_limits)
room.add_calc_zone(eye_limits)
room.add_lamp(lamp)
room.calculate()


room.to_json('tests/test.json')
newroom = Room.from_json('tests/test.json')
newroom.calculate()