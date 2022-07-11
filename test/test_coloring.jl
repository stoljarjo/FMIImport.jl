using FMIZoo
using FMIImport
using FMIImport: fmi2ProvidesDirectionalDerivative, fmi2GetDirectionalDerivative!, setJacobianEntries!, fmi2SampleDirectionalDerivative!
# using FMIImport: fmi2DependencyKind, fmi2DependencyKindIndependent, fmi2DependencyKindDependent
using FMI
using FMI: fmi2GetDependenciesA
using FMICore
using SparseArrays
using SparseArrays: AbstractSparseMatrixCSC
using Graphs


function fmi2GetJacobianMy(comp::FMU2Component, 
                         rdx::AbstractArray{fmi2ValueReference}, 
                         rx::AbstractArray{fmi2ValueReference}; 
                         steps::Union{AbstractArray{fmi2Real}, Nothing} = nothing)
    if isdefined(comp.fmu, :dependencies)
        mat = spzeros(fmi2Real, length(rdx), length(rx))
    else
        mat = zeros(fmi2Real, length(rdx), length(rx))
    end
    fmi2GetJacobianMy!(mat, comp, rdx, rx; steps=steps)
    return mat
end

function fmi2GetJacobianMy!(jac::AbstractMatrix{fmi2Real}, 
                          comp::FMU2Component, 
                          rdx::AbstractArray{fmi2ValueReference}, 
                          rx::AbstractArray{fmi2ValueReference}; 
                          steps::Union{AbstractArray{fmi2Real}, Nothing} = nothing)

    @assert size(jac) == (length(rdx), length(rx)) ["fmi2GetJacobian!: Dimension missmatch between `jac` $(size(jac)), `rdx` ($length(rdx)) and `rx` ($length(rx))."]

    if length(rdx) == 0 || length(rx) == 0
        return nothing
    end 

    #TODO
    # if isdefined(comp.fmu, :dependencies)
        fmi2GetJacobianDependencyMy!(jac, comp, rdx, rx)
        # fmi2GetJacobianDependencyMy!(jac, comp)
    # else
    #     fmi2GetJacobianNoDependency!(jac, comp, rdx, rx)
    # end
    
    return nothing
end

function fmi2GetJacobianDependencyMy!(jac::AbstractSparseMatrixCSC{fmi2Real, Int64}, 
                                    comp::FMU2Component)
     
    rdx = comp.fmu.modelDescription.derivativeValueReferences
    rx = comp.fmu.modelDescription.stateValueReferences
    
    ddsupported = fmi2ProvidesDirectionalDerivative(comp.fmu)

    partialColoringD2(comp.fmu; coloringType=:columns)

    directionalDerivatives = zeros(fmi2Real, length(rdx))
    I, J, _ = findnz(comp.fmu.dependencies)

    for color in unique(comp.fmu.colors)
        indices = findall(==(true), comp.fmu.colors .== color)
        
        if ddsupported
            fill!(directionalDerivatives, zero(fmi2Real))
            fmi2GetDirectionalDerivative!(comp, rdx, rx[indices], directionalDerivatives)
            setJacobianEntries!(jac, I, J, directionalDerivatives, indices)
        else 
            fmi2SampleDirectionalDerivative!(comp, rdx, rx[indices], @view(jac[:, indices]))
        end
    end
    nothing
end

function fmi2GetJacobianDependencyMy!(jac::AbstractSparseMatrixCSC{fmi2Real, Int64}, 
                                    comp::FMU2Component, 
                                    rdx::AbstractArray{fmi2ValueReference},
                                    rx::AbstractArray{fmi2ValueReference})
     
    ddsupported::Bool = fmi2ProvidesDirectionalDerivative(comp.fmu)
    
    # 1: check if rdx and rx are entries for the whole dependency matrix
    if size(comp.fmu.dependencies) == (length(rdx), length(rx))
        fmi2GetJacobianDependencyMy!(jac, comp)
        return nothing
    end

    # 2: get indices for sub arrays
    rdxIndices::Vector{Int64} = [Int64(comp.fmu.modelDescription.derivativeReferenceIndicies[key]) for key in rdx]
    rxIndices::Vector{Int64} = [Int64(comp.fmu.modelDescription.stateReferenceIndicies[key]) for key in rx]

    # 3: select update type
    # check if jac has entries at the indices  
    if nnz(jac) == nnz(comp.fmu.dependencies[rdxIndices][rxIndices])
        updateType = :dependent
    else
        # check if the values for update type 1 or 2 are missing
        #TODO fix dependencyKind not Symbol
        if any(comp.fmu.dependencies[rdxIndices][rxIndices] .∈ [fmi2DependencyKindConstant, fmi2DependencyKindFixed])
            updateType = :constant
        elseif any(comp.fmu.dependencies[rdxIndices][rxIndices] .∈ [fmi2DependencyKindTunable, fmi2DependencyKindDiscrete])
            updateType = :tunable
        else
            updateType = :dependent
        end
    end

    # 4: update the coloring for the new graph
    updateColoring!(comp.fmu; updateType=updateType, rdxIndices=rdxIndices, 
                    rxIndices=rxIndices, coloringType=:columns)

    directionalDerivatives = zeros(fmi2Real, length(rdx))
    I, J, _ = findnz(comp.fmu.dependencies)
    # 5: get derivatives for same colors at onces
    for color in unique(comp.fmu.colors)
        indices = findall(==(true), comp.fmu.colors .== color)
        
        if ddsupported
            fill!(directionalDerivatives, zero(fmi2Real))
            fmi2GetDirectionalDerivative!(comp, rdx, rx[indices], directionalDerivatives)
################ TODO
            setJacobianEntries!(jac, I, J, directionalDerivatives, indices)
        else 
            fmi2SampleDirectionalDerivative!(comp, rdx, rx[indices], @view(jac[:, indices]))
        end
    end
    nothing
end

##### EXAMPLES

# myFMU = fmiLoad("BouncingBall1D", "Dymola", "2022x")
# myFMU = fmiLoad("SpringPendulum1D", "Dymola", "2022x")
myFMU = fmiLoad("SpringFrictionPendulum1D", "Dymola", "2022x")
# myFMU = fmi2Load("./../hiwi_stoljarjohannes/FMUs/GridCabinAcceleration_Examples_Case3d_0FixedDistribution_0CoolDown_ME_12.fmu")
# myFMU = fmi2Load("./../hiwi_stoljarjohannes/FMUs/GridCabinAcceleration_Examples_Case3d_0FixedDistribution_0CoolDown_ME_grid.fmu")

comp = fmi2Instantiate!(myFMU; loggingOn=true)
fmi2SetupExperiment(comp)
fmi2EnterInitializationMode(comp)
fmi2ExitInitializationMode(comp)

@timev depMat = fmi2GetDependenciesA(myFMU)

# rdx = myFMU.modelDescription.derivativeValueReferences
# rx = myFMU.modelDescription.stateValueReferences
# keys = [Int64(myFMU.modelDescription.derivativeReferenceIndicies[key]) for key in rdx[[1, 10, 150]]]

# directionalDerivatives = ones(3)
# fmi2GetDirectionalDerivative!(comp, rdx[[1, 10, 150]], rx[[10, 66]], directionalDerivatives)
# directionalDerivatives

jac = fmi2GetJacobianMy(comp, [myFMU.modelDescription.derivativeValueReferences[2]], [myFMU.modelDescription.stateValueReferences[1]])

# print graph
using GraphPlot, Compose
import Cairo, Fontconfig

g = comp.fmu.graph
nodelabel = 1:nv(g)
draw(PNG("./wheel10.png", 16cm, 16cm), gplot(g, nodelabel=nodelabel))

rem_vertices!(g, [2], keep_order=false)
nodelabel = 1:nv(g)
draw(PNG("./wheel11.png", 16cm, 16cm), gplot(g, nodelabel=nodelabel))

jac2 = fmi2GetJacobianMy!(jac, comp, myFMU.modelDescription.derivativeValueReferences, myFMU.modelDescription.stateValueReferences)
jac3 = fmi2GetFullJacobian(comp, myFMU.modelDescription.derivativeValueReferences, myFMU.modelDescription.stateValueReferences)

# Check Jacobians
# jac ≈ jac3

println("Start colors: $(comp.fmu.colors)")
for dependType in [:all] #, :fixed, :tunable, :discrete, :all, :independent]
    colors = updateColoring!(comp.fmu; updateType=dependType, coloringType=:columns)
    println("$(dependType):  Colors: $colors")
end

println("End colors: $(comp.fmu.colors)")

fmiUnload(myFMU)

println("End of Testbench")
