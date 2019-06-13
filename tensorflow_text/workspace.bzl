"""doc"""

load("//third_party/icu:workspace.bzl", icu = "repo")

def initialize_third_party_archives():
    icu()
