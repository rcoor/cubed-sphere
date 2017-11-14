from .CartesianHighresModel import CartesianHighres
from .CartesianModel import CartesianModel
from .CubedSphereBandedModel import CubedSphereBandedModel
from .CubedSphereDenseModel import CubedSphereDenseModel
from .CubedSphereModel import CubedSphereModel
from .CubedSphereBandedDisjointModel import CubedSphereBandedDisjointModel
from .CubedSphereHighRModel import CubedSphereHighRModel
from .SphericalBandedDisjointModel import SphericalBandedDisjointModel
from .SphericalModel import SphericalModel
from .SphericalHighRModel import SphericalHighRModel

models = {"CubedSphereModel": CubedSphereModel,
          "CubedSphereBandedModel": CubedSphereBandedModel,
          "CubedSphereBandedDisjointModel": CubedSphereBandedDisjointModel,
          "CubedSphereDenseModel": CubedSphereDenseModel,
          "SphericalModel": SphericalModel,
          "SphericalHighRModel": SphericalHighRModel,
          "SphericalBandedDisjointModel": SphericalBandedDisjointModel,
          "CubedSphereHighRModel": CubedSphereHighRModel,
          "CartesianModel": CartesianModel,
          "CartesianHighresModel": CartesianHighres}
