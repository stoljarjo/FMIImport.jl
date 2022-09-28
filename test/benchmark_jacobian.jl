############################# Testbench #####################################
using FMIZoo
using FMICore
using FMIImport
using FMI: fmi2GetDependenciesA
# using LightGraphs
# using LightGraphs: AbstractGraph
# using SparseArrays: spzeros, findnz
using BenchmarkTools

function fmi2GetJacobianOld(comp::FMU2Component, 
                         rdx::AbstractArray{fmi2ValueReference}, 
                         rx::AbstractArray{fmi2ValueReference}; 
                         steps::Union{AbstractArray{fmi2Real}, Nothing} = nothing)
    mat = zeros(fmi2Real, length(rdx), length(rx))
    fmi2GetJacobianOld!(mat, comp, rdx, rx; steps=steps)
    return mat
end

function fmi2GetJacobianOld!(jac::AbstractMatrix{fmi2Real}, 
                          comp::FMU2Component, 
                          rdx::AbstractArray{fmi2ValueReference}, 
                          rx::AbstractArray{fmi2ValueReference}; 
                          steps::Union{AbstractArray{fmi2Real}, Nothing} = nothing)

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

function fmi2GetJacobianFull(comp::FMU2Component, 
                             rdx::AbstractArray{fmi2ValueReference}, 
                             rx::AbstractArray{fmi2ValueReference}; 
                             steps::Union{AbstractArray{fmi2Real}, Nothing} = nothing)
    mat = zeros(fmi2Real, length(rdx), length(rx))
    fmi2GetJacobianFull!(mat, comp, rdx, rx; steps=steps)
    return mat
end

function fmi2GetJacobianFull!(jac::AbstractMatrix{fmi2Real}, 
                              comp::FMU2Component, 
                              rdx::AbstractArray{fmi2ValueReference}, 
                              rx::AbstractArray{fmi2ValueReference}; 
                              steps::Union{AbstractArray{fmi2Real}, Nothing} = nothing)
    @assert size(jac) == (length(rdx),length(rx)) "fmi2GetFullJacobian!: Dimension missmatch between `jac` $(size(jac)), `rdx` ($length(rdx)) and `rx` ($length(rx))."

    if length(rdx) == 0 || length(rx) == 0
        jac = zeros(length(rdx), length(rx))
        return nothing
    end 

    if fmi2ProvidesDirectionalDerivative(comp.fmu)
        for i in 1:length(rx)
            jac[:,i] = fmi2GetDirectionalDerivative(comp, rdx, [rx[i]])
        end
    else
        jac = fmi2SampleDirectionalDerivative(comp, rdx, rx)
    end

    return nothing
end

myFMUs = Dict(
    "BouncingBall1D"               => ("BouncingBall1D", "Dymola", "2022x"),
    "SpringPendulum1D"             => ("SpringPendulum1D", "Dymola", "2022x"),
    "SpringFrictionPendulum1D"     => ("SpringFrictionPendulum1D", "Dymola", "2022x"),
    "SpringDamperPendulum1D"       => ("SpringDamperPendulum1D", "Dymola", "2022x"),
    "SpringTimeFrictionPendulum1D" => ("SpringTimeFrictionPendulum1D", "Dymola", "2022x"),
    "GridCabinAcceleration_12"     => "./../hiwi_stoljarjohannes/FMUs/GridCabinAcceleration_Examples_Case3d_0FixedDistribution_0CoolDown_ME_12.fmu",
    "GridCabinAcceleration_grid"   => "./../hiwi_stoljarjohannes/FMUs/GridCabinAcceleration_Examples_Case3d_0FixedDistribution_0CoolDown_ME_grid.fmu",
    "Longitudinaldynamic"           => "./../hiwi_stoljarjohannes/Modelica/Longitudinaldynamic_LongitudinaldynamicmodelContinuous.fmu"

)

jac1 = nothing; jac2 = nothing; jac3 = nothing;
time1 = nothing; time2 = nothing; time3 = nothing;
io = IOContext(stdout, :histmin=>0.5, :histmax=>8, :logbins=>true)

for (key, value) in myFMUs
    println(io, "#FMU: $key")
    myFMU = typeof(value) == typeof(value) == Tuple{String, String, String} ? fmi2Load(value...) : fmi2Load(value)
    myFMU.executionConfig.assertOnWarning = true

    comp = fmi2Instantiate!(myFMU; loggingOn=true)
    fmi2SetupExperiment(comp)

    if key == "Longitudinaldynamic"
        relativePath = "../hiwi_stoljarjohannes/Modelica/Data"
        pathToDCTable = joinpath(dirname(@__FILE__), relativePath, "DrivingCycle/WLTP_class_1.mat")
        pathToEDTable = joinpath(dirname(@__FILE__), relativePath, "ElectricDriveData.mat")
        pathToPETable = joinpath(dirname(@__FILE__), relativePath, "PowerElectronicsData.mat")
        fmiSetString(myFMU, "dcFileName", pathToDCTable)
        fmiSetString(myFMU, "edFileName", pathToEDTable)
        fmiSetString(myFMU, "peFileName", pathToPETable)
    end

    fmi2EnterInitializationMode(comp)
    fmi2ExitInitializationMode(comp)
    
    time1 = @benchmark (
        fmi2GetJacobianOld(comp, myFMU.modelDescription.derivativeValueReferences, myFMU.modelDescription.stateValueReferences)
    )
    jac1 = fmi2GetJacobianOld(comp, myFMU.modelDescription.derivativeValueReferences, myFMU.modelDescription.stateValueReferences)
        
    time2 = @benchmark (
        fmi2GetJacobianFull(comp, myFMU.modelDescription.derivativeValueReferences, myFMU.modelDescription.stateValueReferences)
    )
    jac2 = fmi2GetJacobianFull(comp, myFMU.modelDescription.derivativeValueReferences, myFMU.modelDescription.stateValueReferences)


    time3 = @benchmark (
        fmi2GetDependenciesA(comp.fmu);
        fmi2GetJacobian(comp, myFMU.modelDescription.derivativeValueReferences, myFMU.modelDescription.stateValueReferences)
    )
    jac3 = fmi2GetJacobian(comp, myFMU.modelDescription.derivativeValueReferences, myFMU.modelDescription.stateValueReferences)

    maxColors = maximum(comp.fmu.colors)
    
    println("\n-Old implementation:")
    println("")
    show(io, MIME("text/plain"), time1)
    println("")
    println("\n---------------------------------------------------------------- \n-Test implementation:")
    show(io, MIME("text/plain"), time2)
    println("")
    println("\n---------------------------------------------------------------- \n-Color implementation:")
    show(io, MIME("text/plain"), time3)
    println("")
    println("Max colors:\t $maxColors \n
    Checks: 
        Jac1 = Jac2: $(jac1 ≈ jac2)
        Jac1 = Jac3: $(jac1 ≈ jac3)
        Jac2 = Jac3: $(jac2 ≈ jac3)
    \n________________________________________________________________")

    fmi2Unload(myFMU)
end

println("END")
