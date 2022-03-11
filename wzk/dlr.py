import os

# USERNAME = os.path.expanduser("~").split(sep='/')[-1]
# TODO is this smarter static or dynamic
USERNAME = "tenh_jo"


def where_am_i():
    location_dict = dict(jote='mac',
                         tenh_jo='dlr',
                         johannes_tenhumberg_gmail_com='gcp')
    try:
        location = location_dict[USERNAME]
    except KeyError:
        location = 'dlr'

    return location


LOCATION = where_am_i()

# Alternative storage places for the samples
DLR_USERSTORE = f"/volume/USERSTORE/{USERNAME}"  # Daily Back-up, relies on connection -> not for large Measurements
DLR_HOMELOCAL = f"/home_local/{USERNAME}"        # No Back-up, but fastest drive -> use for calculation
DLR_USB = f"/var/run/media/{USERNAME}/DLR-MA"


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
