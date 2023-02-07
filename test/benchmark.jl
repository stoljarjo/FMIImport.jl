#
# Copyright (c) 2021 Tobias Thummerer, Lars Mikelsons
# Licensed under the MIT license. See LICENSE file in the project root for details.
#
using FMIZoo
using FMI
using FMI: fmi2GetDependenciesA

import FMIImport: fmi2ProvidesDirectionalDerivative, fmi2GetDirectionalDerivative!, fmi2JVP!, init_jacobian!, eventOccurred!
using FMICore
using SparseArrays: spzeros
using ForwardDiff: jacobian
using BenchmarkTools


tStart = 0.0
tStop = 1.0


# generate training data
# realFMU = fmiLoad("SpringFrictionPendulum1D", "Dymola", "2022x")  # myFMU = fmiLoad("SpringPendulum1D", "Dymola", "2022x")

# realFMU = fmiLoad("./../hiwi_stoljarjohannes/Modelica/Longitudinaldynamic_LongitudinaldynamicmodelContinuous.fmu";  type=fmi2TypeModelExchange)

# Grid
# realFMU = fmiLoad("./../hiwi_stoljarjohannes/FMUs/GridCabinAcceleration_Examples_Case3d_0FixedDistribution_0CoolDown_ME_grid.fmu";  type=fmi2TypeModelExchange)

# Grid 12
realFMU = fmiLoad("./../hiwi_stoljarjohannes/FMUs/GridCabinAcceleration_Examples_Case3d_0FixedDistribution_0CoolDown_ME_12.fmu";  type=fmi2TypeModelExchange)



realFMU.executionConfig.externalCallbacks = false
realFMU.executionConfig.loggingOn = false
realFMU.executionConfig.JVPBuiltInDerivatives = true
realFMU.type = fmi2TypeModelExchange

fmiInstantiate!(realFMU; loggingOn=false)
fmiSetupExperiment(realFMU, tStart, tStop)

# if occursin("Longitudinaldynamic", realFMU.modelName) 
#     relativePath = "../../hiwi_stoljarjohannes/Modelica/Data"
#     pathToDCTable = normpath(joinpath(dirname(@__FILE__), relativePath, "DrivingCycle/WLTP_class_2.mat"))
#     pathToEDTable = normpath(joinpath(dirname(@__FILE__), relativePath, "ElectricDriveData.mat"))
#     pathToPETable = normpath(joinpath(dirname(@__FILE__), relativePath, "PowerElectronicsData.mat"))    
#     fmiSetString(realFMU, "dcFileName", pathToDCTable)
#     fmiSetString(realFMU, "edFileName", pathToEDTable)
#     fmiSetString(realFMU, "peFileName", pathToPETable)
# end

fmiEnterInitializationMode(realFMU)
fmiExitInitializationMode(realFMU)

x0 = fmiGetContinuousStates(realFMU)
numStates = length(x0)


# set solution instance 
component = realFMU.components[end]
component.solution = FMU2Solution(component)


_f = _x -> realFMU(;x=_x)[2]
_f(x0)

# Orig
time = @benchmark ( j = jacobian(_f, x0) )
numEvals = realFMU.components[end].solution.evals_∂ẋ_∂x

realFMU.components[end].solution.evals_∂ẋ_∂x = 0
time1 = @benchmark (
    realFMU.components[end].A.valid = false;
    j = jacobian(_f, x0)    
)
numEvals = realFMU.components[end].solution.evals_∂ẋ_∂x

# Saving
realFMU.executionConfig.JVPBuiltInDerivatives = false
realFMU.components[end].A.valid = false;

realFMU.components[end].solution.evals_∂ẋ_∂x = 0
time = @benchmark ( j = jacobian(_f, x0) )
numEvals = realFMU.components[end].solution.evals_∂ẋ_∂x

realFMU.components[end].solution.evals_∂ẋ_∂x = 0
time1 = @benchmark ( 
    realFMU.components[end].A.valid = false;
    j = jacobian(_f, x0)    
)
numEvals = realFMU.components[end].solution.evals_∂ẋ_∂x


# Saving + Dependencies
fmi2GetDependenciesA(realFMU)
@timev j = jacobian(_f, x0)
realFMU.components[end].A.valid = false;
eventOccurred!(realFMU; eventType= :all)

realFMU.components[end].solution.evals_∂ẋ_∂x = 0
time = @benchmark ( j = jacobian(_f, x0) )
numColors = length(unique(realFMU.colors))
numEvals = realFMU.components[end].solution.evals_∂ẋ_∂x

realFMU.components[end].solution.evals_∂ẋ_∂x = 0
time1 = @benchmark ( 
    realFMU.components[end].A.valid = false;
    j = jacobian(_f, x0)    
)
numEvals = realFMU.components[end].solution.evals_∂ẋ_∂x


# realFMU.components[end].A.valid = false;
# eventOccurred!(realFMU; eventType= :discrete)

# time = @benchmark ( j = jacobian(_f, x0) )
# numColors = length(unique(realFMU.colors))

# time1 = @benchmark ( 
#     realFMU.components[end].A.valid = false;
#     j = jacobian(_f, x0)    
# )

realFMU.components[end].A.valid = false;
eventOccurred!(realFMU; eventType= :dependent)

realFMU.components[end].solution.evals_∂ẋ_∂x = 0
time = @benchmark ( j = jacobian(_f, x0) )
numColors = length(unique(realFMU.colors))
numEvals = realFMU.components[end].solution.evals_∂ẋ_∂x


realFMU.components[end].solution.evals_∂ẋ_∂x = 0
time1 = @benchmark ( 
    realFMU.components[end].A.valid = false;
    j = jacobian(_f, x0)    
)
numEvals = realFMU.components[end].solution.evals_∂ẋ_∂x


println("END")
