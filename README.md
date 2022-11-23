# CMPUT652FinalProject
As the name suggests

# Instructions on RL

## Virtual environment setup (For RL portion) - TODO: combine with venv for transformer if dependencies dont conflict?
- cd Mario-AI-Framework
- virtualenv venv
- source venv/bin/activate
- do this if running for the first time: pip install -r requirements.txt

*** Edit line 38 of mario_env.py to point to the full path of your java src folder because for now relative path not working! ***
i.e. jpype.addClassPath(".../path/to/folder.../src") # TODO: *** Edit this to point to your own java build, relative pathing not working correctly ***

