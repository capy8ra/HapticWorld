Enable can bus
```
# left follower
sudo ip link set can0 up type can bitrate 1000000 dbitrate 5000000 fd on
# right follower
sudo ip link set can1 up type can bitrate 1000000 dbitrate 5000000 fd on
# right leader
sudo ip link set can2 up type can bitrate 1000000 dbitrate 5000000 fd on
# left leader
sudo ip link set can3 up type can bitrate 1000000 dbitrate 5000000 fd on
```

Install `uv` virtual environment
```
uv sync
uv pip install -e .
```