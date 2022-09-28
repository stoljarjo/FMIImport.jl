#
# Copyright (c) 2021 Tobias Thummerer, Lars Mikelsons
# Licensed under the MIT license. See LICENSE file in the project root for details.
#
import FMIImport: fmi2GetJacobian!, fmi2ProvidesDirectionalDerivative, fmi2GetDirectionalDerivative, fmi2GetDirectionalDerivative!
using FMICore

function fmi2GetJacobian!(jac::AbstractMatrix{fmi2Real}, 
                          comp::FMU2Component, 
                          rdx::Array{fmi2ValueReference}, 
                          rx::Array{fmi2ValueReference}; 
                          steps::Union{Array{fmi2Real}, Nothing} = nothing)

    @assert size(jac) == (length(rdx), length(rx)) ["fmi2GetJacobian!: Dimension missmatch between `jac` $(size(jac)), `rdx` ($length(rdx)) and `rx` ($length(rx))."]

    if length(rdx) == 0 || length(rx) == 0
        jac = zeros(length(rdx), length(rx))
        return nothing
    end 

    ddsupported = fmi2ProvidesDirectionalDerivative(comp.fmu)

    rdx_inds = collect(comp.fmu.modelDescription.valueReferenceIndicies[vr] for vr in rdx)
    rx_inds  = collect(comp.fmu.modelDescription.valueReferenceIndicies[vr] for vr in rx)
    
    for i in 1:length(rx)

        sensitive_rdx_inds = 1:length(rdx)
        sensitive_rdx = rdx

        if length(sensitive_rdx) > 0
            if ddsupported
                fmi2GetDirectionalDerivative!(comp, sensitive_rdx, [rx[i]], view(jac, sensitive_rdx_inds, i))
            else 
                fmi2SampleDirectionalDerivative!(comp, sensitive_rdx, [rx[i]], view(jac, sensitive_rdx_inds, i))

            end
        end
    end
     
    return nothing
end



# function fmi2GetFullJacobian!(jac::AbstractMatrix{fmi2Real}, 
#                               comp::FMU2Component, 
#                               rdx::Array{fmi2ValueReference}, 
#                               rx::Array{fmi2ValueReference}; 
#                               steps::Union{Array{fmi2Real}, Nothing} = nothing)

# function fmi2GetJacobian!(jac::AbstractMatrix{fmi2Real}, 
#                           comp::FMU2Component, 
#                           rdx::Array{fmi2ValueReference}, 
#                           rx::Array{fmi2ValueReference}; 
#                           steps::Union{Array{fmi2Real}, Nothing} = nothing)

#     @assert size(jac) == (length(rdx),length(rx)) "fmi2GetFullJacobian!: Dimension missmatch between `jac` $(size(jac)), `rdx` ($length(rdx)) and `rx` ($length(rx))."

#     # @warn "`fmi2GetFullJacobian!` is for benchmarking only, please use `fmi2GetJacobian`."

#     if length(rdx) == 0 || length(rx) == 0
#         jac = zeros(length(rdx), length(rx))
#         return nothing
#     end 

#     if fmi2ProvidesDirectionalDerivative(comp.fmu)
#         for i in 1:length(rx)
#             jac[:,i] = fmi2GetDirectionalDerivative(comp, rdx, [rx[i]])
#         end
#     else
#         jac = fmi2SampleDirectionalDerivative(comp, rdx, rx)
#     end

#     return nothing
# end


using FMIFlux
using FMIFlux.FMIImport: fmi2StringToValueReference, fmi2ValueReference
using FMIFlux: fmi2GetSolutionState, fmi2GetSolutionValue, fmi2GetSolutionTime
using Test
using FMIZoo
using FMI
using FMI: fmi2GetDependenciesA
using Flux
using DifferentialEquations: Tsit5


methods(fmi2GetJacobian!)

import Random 
Random.seed!(1234);

t_start = 0.0
t_step = 0.01
t_stop = 5.0
tData = t_start:t_step:t_stop

# generate training data
# realFMU = fmiLoad("SpringFrictionPendulum1D", "Dymola", "2022x")
realFMU = fmiLoad("./../hiwi_stoljarjohannes/FMUs/GridCabinAcceleration_Examples_Case3d_0FixedDistribution_0CoolDown_ME_12.fmu")

fmiInstantiate!(realFMU; loggingOn=true)
fmiSetupExperiment(realFMU, t_start, t_stop)
fmiEnterInitializationMode(realFMU)
fmiExitInitializationMode(realFMU)
x0 = fmiGetContinuousStates(realFMU)
realSimData = fmiSimulateCS(realFMU, t_start, t_stop; recordValues=["der(CaseFMU.cabin.cabinIO.cabin.w1001.conduction.T[3])"], setup=false, reset=false, instantiate=false, saveat=tData)

# load FMU for NeuralFMU
# myFMU = fmiLoad("SpringPendulum1D", "Dymola", "2022x")
# myFMU = fmiLoad("./../hiwi_stoljarjohannes/FMUs/GridCabinAcceleration_Examples_Case3d_0FixedDistribution_0CoolDown_ME_12.fmu")
# myFMU = fmiLoad("./../hiwi_stoljarjohannes/FMUs/GridCabinAcceleration_Examples_Case3d_0FixedDistribution_0CoolDown_ME_grid.fmu")
myFMU = fmiLoad("./../hiwi_stoljarjohannes/Modelica/Longitudinaldynamic_LongitudinaldynamicmodelContinuous.fmu")

# myFMU.executionConfig.assertOnWarning = true
myFMU.executionConfig.maxNewDiscreteStateCalls = 100

fmiInstantiate!(myFMU; loggingOn=true)
###### TODO set other function 
# myFMU.components[end].jacobianUpdate! = oldfmi2GetJacobian!
# myFMU.comrelativePath = "../hiwi_stoljarjohannes/Modelica/Data"
pathToDCTable = joinpath(dirname(@__FILE__), relativePath, "DrivingCycle/WLTP_class_1.mat")
pathToEDTable = joinpath(dirname(@__FILE__), relativePath, "ElectricDriveData.mat")
pathToPETable = joinpath(dirname(@__FILE__), relativePath, "PowerElectronicsData.mat")
fmiSetString(myFMU, "dcFileName", pathToDCTable)
fmiSetString(myFMU, "edFileName", pathToEDTable)
fmiSetString(myFMU, "peFileName", pathToPETable)ponents[end].jacobianUpdate! = oldfmi2GetFullJacobian!
# TODO definieren der Kennfelder
# DrivingCycle.mat
# ElectricDriveData.mat
# PowerElectronicsData.mat

fmiSetupExperiment(myFMU, t_start, t_stop)



fmiEnterInitializationMode(myFMU)
fmiExitInitializationMode(myFMU)

fmi2GetDependenciesA(myFMU)

# setup traing data
# posData = fmi2GetSolutionValue(realSimData, "mass.s")
posData = fmi2GetSolutionValue(realSimData, "der(CaseFMU.cabin.cabinIO.cabin.w1001.conduction.T[3])")
posData .+= randn() ./ 10

# loss function for training
function losssum()
    global problem, x0, posData
    solution = problem(x0)

    posNet = fmi2GetSolutionState(solution, 1; isIndex=true)
    # posNet = fmi2GetSolutionState(solution, 5238; isIndex=true)
    
    Flux.Losses.mse(posNet, posData)
end

# callback function for training
global iterCB = 0
global lastLoss = 0.0
function callb()
    global iterCB += 1
    global lastLoss

    if iterCB % 10 == 0
        loss = losssum()
        @info "Loss: $loss"
        @test loss < lastLoss  
        lastLoss = loss
    end
end

numStates = 360

net = Chain(states ->  fmiEvaluateME(myFMU, states), 
            Dense(numStates, 16, tanh),
            Dense(16, 16, tanh),
            Dense(16, numStates))

optim = ADAM(1e-4)
global problem, lastLoss, iterCB
problem = ME_NeuralFMU(myFMU, net, (t_start, t_stop), Tsit5(); saveat=tData)

solutionBefore = problem(x0)

# train it ...
p_net = Flux.params(problem)
length(collect(keys(p_net.params.dict))[1])

iterCB = 0
lastLoss = losssum()
@info "Start-Loss for net: $lastLoss"
@timev Flux.train!(losssum, p_net, Iterators.repeated((), 1), optim; cb=callb)

# check results
solutionAfter = problem(x0)
    

fmiUnload(realFMU)
fmiUnload(myFMU)
