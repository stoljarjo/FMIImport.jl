using Graphs
using Graphs: AbstractGraph
using SparseArrays: findnz, SparseMatrixCSC


const global validUpdateTypes::IdDict{Symbol, Vector{fmi2DependencyKind}} = IdDict(
    :all => [fmi2DependencyKindDependent, fmi2DependencyKindTunable, fmi2DependencyKindDiscrete, fmi2DependencyKindConstant, fmi2DependencyKindFixed],
    :constant => [fmi2DependencyKindDependent, fmi2DependencyKindTunable, fmi2DependencyKindDiscrete, fmi2DependencyKindConstant, fmi2DependencyKindFixed],
    :fixed => [fmi2DependencyKindDependent, fmi2DependencyKindTunable, fmi2DependencyKindDiscrete, fmi2DependencyKindConstant, fmi2DependencyKindFixed],
    :tunable => [fmi2DependencyKindDependent, fmi2DependencyKindTunable, fmi2DependencyKindDiscrete],
    :discrete => [fmi2DependencyKindDependent, fmi2DependencyKindTunable, fmi2DependencyKindDiscrete],
    :dependent => [fmi2DependencyKindDependent],
    :independent => []
)


"""
Create a SimpleGraph out of the dependency matrix.
"""
function createGraph(fmu::FMU2) ::SimpleGraph
    if !isdefined(fmu, :dependencies)
        @warn "Dependencies are not defined!"
        return SimpleGraph()
    end
    I, J, _ = findnz(fmu.dependencies)
    addOffsetToVertex!(J, fmu.dependencies)
   
    fmu.graph = SimpleGraph(sum(size(fmu.dependencies)))
    for (v1, v2) in zip(I, J)
        add_edge!(fmu.graph, v1, v2)
    end
    fmu.graph
end

function updateGraph(fmu::FMU2; updateType::Symbol) ::SimpleGraph
    if updateType === :independent
        return SimpleGraph(nv(fmu.graph))
    end
    
    graph = SimpleGraph(fmu.graph)
    if updateType ∉ [:all, :constant, :fixed]
        I, J, V = findnz(fmu.dependencies)
        addOffsetToVertex!(J, fmu.dependencies)
        dependencyTypes = selectDependencyTypes(updateType)
        for (v1, v2, value) in zip(I, J, V)
            if value ∉ dependencyTypes
                rem_edge!(graph, v1, v2)
            end
        end
    end
    return graph
end
function updateGraph(dependencies::SparseMatrixCSC{Union{Nothing, fmi2DependencyKind}, Int64}; updateType::Symbol) ::SimpleGraph
    I, J, V = findnz(dependencies)
    addOffsetToVertex!(J, dependencies)
    dependencyTypes = selectDependencyTypes(updateType)
    
    graph = SimpleGraph(sum(size(dependencies)))
    for (v1, v2, value) in zip(I, J, V)
        if value ∈ dependencyTypes 
            add_edge!(graph, v1, v2)
        end
    end
    return graph
end

function selectDependencyTypes(updateType::Symbol) ::Vector{fmi2DependencyKind}
    if !haskey(validUpdateTypes, updateType)
        @warn "Undefined update type"
        return []
    end

    return validUpdateTypes[updateType]
end

function addOffsetToVertex!(vertexIndices::AbstractVector{Int64}, dependencies::AbstractMatrix{Union{fmi2DependencyKind, Nothing}})
    # add offset to J vertex
    (numRows, _)  = size(dependencies)
    vertexIndices .+= numRows
    return nothing
end

function getVertices(dimDependencies::Tuple{Int64, Int64}; coloringType::Symbol) ::UnitRange{Int64}
    (numRows, numColumns) = dimDependencies
    if coloringType == :rows
        return 1:numRows
    elseif coloringType == :columns
        return numRows+1:numRows+numColumns
    elseif coloringType == :all
        return 1:numRows+numColumns
    else
        @warn "getVertices: Unsupported coloringType= $(String(coloringType))"
    end
end

"""
Reference: 
  What Color Is Your Jacobian? Graph Coloring for Computing Derivatives
  Assefaw Hadish Gebremedhin, Fredrik Manne and Alex Pothen
  https://www.jstor.org/stable/20453700
  ALGORITHM 3.2. A greedy partial distance-2 coloring algorithm.

  ==> fast but biggest coloring
"""
function partialColoringD2(fmu::FMU2; coloringType::Symbol=:columns) ::AbstractVector
    if !isdefined(fmu, :graph)
        createGraph(fmu)
    end

    numColors = length(getVertices(size(fmu.dependencies); coloringType=coloringType))

    if !isdefined(fmu, :colors) || fmu.colors != numColors || fmu.colorType !== coloringType 
        fmu.colorType = coloringType
        fmu.colors = partialColoringD2(fmu.graph, size(fmu.dependencies); coloringType=coloringType)
    end
    fmu.colors
end
function partialColoringD2(g::AbstractGraph, dimDependencies::Tuple{Int64, Int64}; coloringType::Symbol) ::AbstractVector
    @info "partialColoringD2: Start partial graph coloring (type: $coloringType)."
    
    vertices = getVertices(dimDependencies; coloringType=coloringType)
    
    num_all_vertices = nv(g)
    forbidden_colors = zeros(Int, num_all_vertices)
    colorvec = zeros(Int, num_all_vertices)
    
    for v in vertices
        for w in neighbors(g, v)
            for x in neighbors(g, w)
                if colorvec[x] != 0
                    forbidden_colors[colorvec[x]] = v
                end
            end
        end
        colorvec[v] = getMinColor(forbidden_colors, v)
    end
    colorvec[vertices]
end

"""
Reference: 
  What Color Is Your Jacobian? Graph Coloring for Computing Derivatives
  Assefaw Hadish Gebremedhin, Fredrik Manne and Alex Pothen
  https://www.jstor.org/stable/20453700
  ALGORITHM 4.1. A greedy star coloring algorithm.

  ==> slow but smaller coloring
"""
function starColoringD2Alg1(fmu::FMU2; coloringType::Symbol=:columns) ::AbstractVector
    if !isdefined(fmu, :graph)
        createGraph(fmu)
    end
    
    numColors = length(getVertices(size(fmu.dependencies); coloringType=coloringType))

    if !isdefined(fmu, :colors) || fmu.colors != numColors || fmu.colorType !== coloringType 
        fmu.colorType = coloringType
        fmu.colors = starColoringD2Alg1(fmu.graph, size(fmu.dependencies); coloringType=coloringType)
    end
    fmu.colors
end
function starColoringD2Alg1(g::AbstractGraph, dimDependencies::Tuple{Int64, Int64}; coloringType::Symbol) ::AbstractVector
    @info "starColoringD2Alg1: Start star graph coloring (algorithm 1) (type: $coloringType)."

    vertices = getVertices(dimDependencies; coloringType=coloringType)

    num_all_vertices = nv(g)
    colorvec = zeros(Int, num_all_vertices)
    forbidden_colors = zeros(Int, num_all_vertices)

    for vertex_i in 1:vertices
        for w in inneighbors(g, vertex_i)
            if colorvec[w] != 0
                forbidden_colors[colorvec[w]] = vertex_i
            end

            for x in inneighbors(g, w)
                if colorvec[x] == 0
                    continue
                end
                if colorvec[w] == 0
                    forbidden_colors[colorvec[x]] = vertex_i
                else
                    for y in inneighbors(g, x)
                        if colorvec[y] == 0
                            continue
                        end
                        if y != w && colorvec[y] == colorvec[w]
                            forbidden_colors[colorvec[x]] = vertex_i
                            break
                        end
                    end
                end
            end
        end
        colorvec[vertex_i] = getMinColor(forbidden_colors, vertex_i)
    end
    colorvec[vertices]
end

"""
Reference: 
  What Color Is Your Jacobian? Graph Coloring for Computing Derivatives
  Assefaw Hadish Gebremedhin, Fredrik Manne and Alex Pothen
  https://www.jstor.org/stable/20453700
  ALGORITHM 4.2. A second greedy star coloring algorithm.

  ==> trade of between execution time and coloring
"""
function starColoringV2Alg2(fmu::FMU2; coloringType::Symbol=:columns) ::AbstractVector
    if !isdefined(fmu, :graph)
        createGraph(fmu)
    end

    numColors = length(getVertices(size(fmu.dependencies); coloringType=coloringType))

    if !isdefined(fmu, :colors) || fmu.colors != numColors || fmu.colorType !== coloringType 
        fmu.colorType = coloringType
        fmu.colors = starColoringV2Alg2(fmu.graph, size(fmu.dependencies); coloringType=coloringType)
    end
    fmu.colors
end
function starColoringV2Alg2(g::AbstractGraph, dimDependencies::Tuple{Int64, Int64}; coloringType::Symbol) ::AbstractVector
    @info "starColoringV2Alg2: Start star graph coloring (algorithm 2) (type: $coloringType)."

    vertices = getVertices(dimDependencies; coloringType=coloringType)

    num_all_vertices = nv(g)
    colorvec = zeros(Int, num_all_vertices)
    forbidden_colors = zeros(Int, num_all_vertices)

    for vertex_i in 1:vertices
        for w in inneighbors(g, vertex_i)
            if colorvec[w] != 0
                forbidden_colors[colorvec[w]] = vertex_i
            end

            for x in inneighbors(g, w)
                if colorvec[x] == 0
                    continue
                end
                if colorvec[w] == 0
                    forbidden_colors[colorvec[x]] = vertex_i
                else
                    if colorvec[x] < colorvec[w]
                        forbidden_colors[colorvec[x]] = vertex_i
                    end
                end
            end
        end
        colorvec[vertex_i] = getMinColor(forbidden_colors, vertex_i)
    end
    colorvec[vertices]
end

"""
Returns the minimal color index.
"""
function getMinColor(forbidden_colors::AbstractVector, vertex_i::Integer) ::Int
    c = 1
    while (forbidden_colors[c] == vertex_i)
        c+=1
    end
    c
end

function updateColoring!(fmu::FMU2; updateType::Symbol, coloringType::Symbol=:columns) ::AbstractVector
    if !isdefined(fmu, :graph)
        createGraph(fmu)
    end

    numColors = length(getVertices(size(fmu.dependencies); coloringType=coloringType))
    if isdefined(fmu, :colors) && length(fmu.colors) == numColors && fmu.colorType == coloringType && fmu.updateType == updateType 
        return fmu.colors
    end

    graph = updateGraph(fmu; updateType=updateType)
    
    fmu.colorType = coloringType
    fmu.updateType = updateType
    fmu.colors = partialColoringD2(graph, size(fmu.dependencies); coloringType=coloringType)
    return fmu.colors
end
function updateColoring!(fmu::FMU2, dependencies::AbstractMatrix{fmi2DependencyKind}; 
                        updateType::Symbol, coloringType::Symbol=:columns) ::AbstractVector

    graph = updateGraph(dependencies; updateType=updateType)
    
    fmu.colorType = coloringType
    fmu.colors = partialColoringD2(graph, size(dependencies); coloringType=coloringType)
    return fmu.colors
end
