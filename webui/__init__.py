# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
ConanAI SAM WebUI — NanoOWL + NanoSAM live demo.

Forked from vlm2 (in-house build on NVIDIA's live-vlm-webui, Apache 2.0). The
WebRTC/aiohttp plumbing, local-file/RTSP tracks, and frontend chrome are preserved;
the VLM inference module is replaced with an in-process NanoOWL + NanoSAM service
(owl_sam_service). See NOTICE.txt for credits.
"""

__version__ = "0.1.0"
__author__ = "ConanAI / jetson-sam-demo"
__license__ = "Apache-2.0"

from . import server
from . import video_processor
from . import gpu_monitor
from . import owl_sam_service
from . import local_file_track
from . import rtsp_track

__all__ = [
    "server",
    "video_processor",
    "gpu_monitor",
    "owl_sam_service",
    "local_file_track",
    "rtsp_track",
]
