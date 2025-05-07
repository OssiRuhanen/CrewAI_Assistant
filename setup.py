from setuptools import setup, find_packages

setup(
    name="agent_assistant",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "crewai",
        "openai",
        "python-dotenv",
        "sounddevice",
        "numpy",
        "scipy",
        "soundfile",
        "google-cloud-texttospeech",
        "pyyaml"
    ],
    python_requires=">=3.8",
) 