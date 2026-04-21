from setuptools import setup, find_packages

setup(
    name='antimalarialtraitpredictions_samples',
    url='https://github.com/alrichardbollans/antimalarial_trait_predictions/tree/sampling',
    author='Adam Richard-Bollans',
    author_email='38588335+alrichardbollans@users.noreply.github.com',
    # Needed to actually package something
    packages=find_packages(),
    install_requires=[
        "automatchnames >= 1.2.3",
        "ApmTraits == 1.0"
    ],
    # *strongly* suggested for sharing
    version='0.1',
    license='MIT',
    long_description=open('README.md').read(),
)
