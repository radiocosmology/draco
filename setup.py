from setuptools import setup, find_packages


setup(
    name = 'ch_pipeline',
    version = 0.1,

    packages = find_packages(),
    author = "CHIME collaboration",
    author_email = "jrs65@cita.utoronto.ca",
    description = "CHIME Pipeline",
    url = "http://bitbucket.org/chime/ch_pipeline/",

    package_data = { "ch_pipeline" : [ "data/*" ] }
)