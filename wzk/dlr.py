import os
import platform


PLATFORM_IS_LINUX = platform.system() == "Linux"

USERNAME = os.path.expanduser("~").split(sep="/")[-1]
TENH_JO = "tenh_jo"

USERNAME_DICT = dict(mac="jote",
                     dlr="tenh_jo",
                     gcp="johannes_tenhumberg_gmail_com")

LOCATION_DICT = dict(jote="mac",
                     tenh_jo="dlr",
                     baeuml="dlr",
                     bauml="dlr",
                     johannes_tenhumberg_gmail_com="gcp")


def where_am_i():
    try:
        location = LOCATION_DICT[USERNAME]
    except KeyError:
        location = "dlr"

    return location


LOCATION = where_am_i()


# Alternative storage places for the samples
DLR_USERSTORE = f"/volume/USERSTORE/{USERNAME}"  # Daily Back-up, relies on connection -> not for large Measurements
DLR_HOMELOCAL = f"/home_local/{USERNAME}"        # No Back-up, but fastest drive -> use for calculation
DLR_HOME = f"/home/{USERNAME}"
DLR_USB = "/var/run/media/tenh_jo/DLR-MA"

TENH_USERSTORE = f"/volume/USERSTORE/{TENH_JO}"

# Paper
if PLATFORM_IS_LINUX:
    DIR_PAPER = f"{DLR_USERSTORE}/Paper"
else:
    DIR_PAPER = "/Users/jote/Documents/PhD/paper"


Humanoids20_Calibration = DIR_PAPER + "/20Humanoids_Calibration"
Humanoids22_Calibration = DIR_PAPER + "/22Humanoids_Calibration"
IROS22_PLANNING = DIR_PAPER + "/22IROS_Planning"
IROS23_IK = DIR_PAPER + "/23IROS_IK"
IROS23_Representation = "/23IROS_Representation"
TRO23_Planning = "/23TRO_Planning"

# Projects
__automatica22_dict = dict(dlr="/volume/USERSTORE/tenh_jo/Automatica2022",
                           mac="/Users/jote/Documents/PhD/data/mogen/Automatica2022",
                           gcp="/home/johannes_tenhumberg_gmail_com/sdb/Automatica2022")
Automatica22 = __automatica22_dict[LOCATION]


# ----------------------------------------------------------------------------------------------------------------------
# DLR Remote Access

# VNC
# On remote PC (i.e. pandia):
#   vncpasswd  -> set password (optional)
#   vncserver -geometry 1680x1050 -depth 24 -> start server, with scaled resolution for mac
#
# On local PC (i.e. my laptop):
#   ssh -l tenh_jo -L 5901:pandia:5901 ssh.robotic.dlr.de  -> set link port to the remote display
#   open VNC and connect to home: 127.0.0.1:1

# Connect2
# ssh -D 8080 -l tenh_jo ssh.robotic.dlr.de
