from .tokenrig_setup import TokenRigSetup
from .tokenrig_generate import TokenRigGenerate
from .tokenrig_load_model import TokenRigLoadModel

NODE_CLASS_MAPPINGS = {
    "TokenRigSetup": TokenRigSetup,
    "TokenRigLoadModel": TokenRigLoadModel,
    "TokenRigGenerate": TokenRigGenerate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TokenRigSetup": "TokenRig Setup (Install Worker)",
    "TokenRigLoadModel": "TokenRig Load Model",
    "TokenRigGenerate": "TokenRig Generate Rig",
}
