#
# Copyright (c) 2021 Tobias Thummerer, Lars Mikelsons
# Licensed under the MIT license. See LICENSE file in the project root for details.
#
using FMIZoo
using FMICore
import FMIImport: fmi2GetJacobian!, fmi2ProvidesDirectionalDerivative, fmi2GetDirectionalDerivative, fmi2GetDirectionalDerivative!, fmi2StringToValueReference, fmi2ValueReference
using FMI
using FMI: fmi2GetDependenciesA
# using FMIFlux
# using FMIFlux: fmi2GetSolutionState, fmi2GetSolutionValue, fmi2GetSolutionTime
# using Flux
# using DifferentialEquations: Tsit5
using Test
import Random 
Random.seed!(1234)

# methods(fmi2GetJacobian!)

# function fmi2GetJacobian!(jac::AbstractMatrix{fmi2Real}, 
#                           comp::FMU2Component, 
#                           rdx::Array{fmi2ValueReference}, 
#                           rx::Array{fmi2ValueReference}; 
#                           steps::Union{Array{fmi2Real}, Nothing} = nothing)

#     @assert size(jac) == (length(rdx), length(rx)) ["fmi2GetJacobian!: Dimension missmatch between `jac` $(size(jac)), `rdx` ($length(rdx)) and `rx` ($length(rx))."]

#     if length(rdx) == 0 || length(rx) == 0
#         jac = zeros(length(rdx), length(rx))
#         return nothing
#     end 

#     ddsupported = fmi2ProvidesDirectionalDerivative(comp.fmu)

#     rdx_inds = collect(comp.fmu.modelDescription.valueReferenceIndicies[vr] for vr in rdx)
#     rx_inds  = collect(comp.fmu.modelDescription.valueReferenceIndicies[vr] for vr in rx)
    
#     for i in 1:length(rx)

#         sensitive_rdx_inds = 1:length(rdx)
#         sensitive_rdx = rdx

#         if length(sensitive_rdx) > 0
#             if ddsupported
#                 fmi2GetDirectionalDerivative!(comp, sensitive_rdx, [rx[i]], view(jac, sensitive_rdx_inds, i))
#             else 
#                 fmi2SampleDirectionalDerivative!(comp, sensitive_rdx, [rx[i]], view(jac, sensitive_rdx_inds, i))

#             end
#         end
#     end
     
#     return nothing
# end

# methods(fmi2GetJacobian!)

tStep = 0.1
tStart = 0.0
tStop = 5.0
tSave = collect(tStart:tStep:tStop)

# generate training data
# realFMU = fmiLoad("SpringFrictionPendulum1D", "Dymola", "2022x")# myFMU = fmiLoad("SpringPendulum1D", "Dymola", "2022x")
# myFMU = fmiLoad("./../hiwi_stoljarjohannes/FMUs/GridCabinAcceleration_Examples_Case3d_0FixedDistribution_0CoolDown_ME_12.fmu")
# myFMU = fmiLoad("./../hiwi_stoljarjohannes/FMUs/GridCabinAcceleration_Examples_Case3d_0FixedDistribution_0CoolDown_ME_grid.fmu")
realFMU = fmiLoad("./../hiwi_stoljarjohannes/Modelica/Longitudinaldynamic_LongitudinaldynamicmodelContinuous.fmu")
fmiInfo(realFMU)

# realFMU.executionConfig.assertOnWarning = true
# realFMU.executionConfig.maxNewDiscreteStateCalls = 100

relativePath = "../../hiwi_stoljarjohannes/Modelica/Data"
pathToDCTable = normpath(joinpath(dirname(@__FILE__), relativePath, "DrivingCycle/WLTP_class_2.mat"))
pathToEDTable = normpath(joinpath(dirname(@__FILE__), relativePath, "ElectricDriveData.mat"))
pathToPETable = normpath(joinpath(dirname(@__FILE__), relativePath, "PowerElectronicsData.mat"))

paramsDict = Dict{String, Any}()
paramsDict["dcFileName"] = pathToDCTable
paramsDict["edFileName"] = pathToEDTable
paramsDict["peFileName"] = pathToPETable
vrs = ["result.vehicleSpeed", "result.cycleSpeed", "result.cumulativeConsumption", "result.consumption"]

manual = true
if manual
    fmiInstantiate!(realFMU; loggingOn=true)
    fmiSetupExperiment(realFMU, tStart, tStop)

    if occursin("Longitudinaldynamic", realFMU.modelName) 
        
        fmiSetString(realFMU, "dcFileName", pathToDCTable)
        fmiSetString(realFMU, "edFileName", pathToEDTable)
        fmiSetString(realFMU, "peFileName", pathToPETable)
    end

    fmiEnterInitializationMode(realFMU)
    fmiExitInitializationMode(realFMU)

    fmi2GetDependenciesA(realFMU)
    # components[end].jacobianUpdate! = oldfmi2GetFullJacobian!

    x0 = fmiGetContinuousStates(realFMU)
    println(x0)
    # realSimData = fmiSimulateME(realFMU, t_start, t_stop; recordValues=["der(CaseFMU.cabin.cabinIO.cabin.w1001.conduction.T[3])"], setup=false, reset=false, instantiate=false, saveat=tData)
    realSimData = fmiSimulateME(realFMU, tStart, tStop; recordValues=vrs, setup=false, reset=false, instantiate=false, saveat=tSave)
else
    fmi2GetDependenciesA(realFMU)
    realSimData = fmiSimulateME(realFMU, tStart, tStop; parameters=paramsDict, recordValues=vrs, saveat=tSave)
end

# realSimData = fmiSimulateME(realFMU, tStart, tStop; parameters=paramsDict, recordValues=vrs, saveat=tSave, setup=true, instantiate=true, freeInstance=true, terminate=true, reset=true)
# realSimData = fmiSimulateCS(realFMU, tStart, tStop; parameters=paramsDict, recordValues=vrs, saveat=tSave, setup=true, instantiate=true, freeInstance=true, terminate=true, reset=true)

# setup traing data
# posData = fmi2GetSolutionValue(realSimData, "der(CaseFMU.cabin.cabinIO.cabin.w1001.conduction.T[3])")
posData = fmi2GetSolutionValue(realSimData, "result.vehicleSpeed")
posData .+= randn() ./ 10

# loss function for training
function losssum(p)
    global neuralFMU, paramsDict, x0, posData
    solution = neuralFMU(x0, tStart; p=p, parameters=paramsDict)
    println(solution)
    # posNet = fmi2GetSolutionState(solution, 5238; isIndex=true)
    posNet = fmi2GetSolutionState(solution, 1; isIndex=true)
  
    println(length(posData))
    println(length(posNet))

    Flux.Losses.mse(posNet, posData)
end

# callback function for training
global iterCB = 0
global lastLoss = 0.0
function callb(p)
    global iterCB += 1
    global lastLoss

    if iterCB % 10 == 0
        loss = losssum(p[1])
        @info "Loss: $loss"
        @test loss < lastLoss  
        lastLoss = loss
    end
end


using FMIFlux
using FMIFlux: fmi2GetSolutionState, fmi2GetSolutionValue, fmi2GetSolutionTime
using Flux
using DifferentialEquations: Tsit5

numStates = length(x0)

net = Chain(states ->  fmiEvaluateME(realFMU, states), 
            Dense(numStates, 16, identity; init=Flux.identity_init)
            Dense(16, numStates, identity; init=Flux.identity_init))

optim = ADAM(1e-6)
global neuralFMU, lastLoss, iterCB
neuralFMU = ME_NeuralFMU(realFMU, net, (tStart, tStop), Tsit5(); saveat=tSave, dt=1e-8)

solutionBefore = neuralFMU(x0; parameters=paramsDict)

# train it ...
p_net = Flux.params(neuralFMU)
length(collect(keys(p_net.params.dict))[1])

iterCB = 0
lastLoss = 0
@info "Start-Loss for net: $lastLoss"
@timev FMIFlux.train!(losssum, p_net, Iterators.repeated((), 11), optim; cb=()->callb(p_net))

# check results
solutionAfter = neuralFMU(x0; parameters=paramsDict)
    
fmiUnload(realFMU)
