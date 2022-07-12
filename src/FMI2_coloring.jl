using Graphs
using Graphs: AbstractGraph
using SparseArrays: findnz

"""
Create a SimpleGraph out of the dependency matrix.
"""
function createGraph(fmu::FMU2) ::SimpleGraph
    if !isdefined(fmu, :dependencies)
        @warn "Dependencies are not defined!"
        return
    end
    I, J, _ = findnz(fmu.dependencies)

    # add offset to J vertex
    num_states = size(fmu.dependencies)[1] 
    J .+= num_states

    fmu.graph = SimpleGraph(num_states * 2)
    for (v1, v2) in zip(I, J)
        add_edge!(fmu.graph, v1, v2)
    end
    fmu.graph
end

function getVertices(num_vertices::Int; coloringType::Symbol) ::UnitRange{Int64}
    num_half_vertices = ceil(Int, num_vertices/2)
    if coloringType == :rows
        return 1:num_half_vertices
    elseif coloringType == :columns
        return num_half_vertices+1:num_vertices
    elseif coloringType == :all
        return 1:num_vertices
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
    if !isdefined(fmu, :colors) || fmu.colorType !== coloringType
        fmu.colorType = coloringType
        fmu.colors = partialColoringD2(fmu.graph; coloringType=coloringType)
    end
    fmu.colors
end
function partialColoringD2(g::AbstractGraph; coloringType::Symbol) ::AbstractVector
    @info "partialColoringD2: Start partial graph coloring."
    
    num_all_vertices = nv(g)
    vertices = getVertices(num_all_vertices; coloringType=coloringType)

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
    if !isdefined(fmu, :colors) || fmu.colorType !== coloringType
        fmu.colorType = coloringType
        fmu.colors = starColoringD2Alg1(fmu.graph; coloringType=coloringType)
    end
    fmu.colors
end
function starColoringD2Alg1(g::AbstractGraph; coloringType::Symbol) ::AbstractVector
    @info "starColoringD2Alg1: Start star graph coloring (algorithm 1)."

    num_all_vertices = nv(g)
    vertices = getVertices(num_all_vertices; coloringType=coloringType)

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
    if !isdefined(fmu, :colors) || fmu.colorType !== coloringType
        fmu.colorType = coloringType
        fmu.colors = starColoringV2Alg2(fmu.graph; coloringType=coloringType)
    end
    fmu.colors
end
function starColoringV2Alg2(g::AbstractGraph; coloringType::Symbol) ::AbstractVector
    @info "starColoringV2Alg2: Start star graph coloring (algorithm 2)."

    num_all_vertices = nv(g)
    vertices = getVertices(num_all_vertices; coloringType=coloringType)

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

    graph = updateGraph(fmu; updateType=updateType)
    
    fmu.colorType = coloringType
    fmu.colors = partialColoringD2(graph; coloringType=coloringType)
    return fmu.colors
end
function updateColoring!(fmu::FMU2, dependencies::AbstractMatrix{fmi2DependencyKind}; 
                        updateType::Symbol, coloringType::Symbol=:columns) ::AbstractVector

    graph = updateGraph(dependencies; updateType=updateType)
    
    fmu.colorType = coloringType
    fmu.colors = partialColoringD2(graph; coloringType=coloringType)
    return fmu.colors
end

function updateGraph(fmu::FMU2; updateType::Symbol) ::SimpleGraph
    graph = SimpleGraph(fmu.graph)
    
    if updateType ∈ [:all, :constant, :fixed]
        return graph

    elseif updateType ∈ [:dependent, :tunable, :discrete]

        I, J, V = findnz(fmu.dependencies)
        num_states = size(fmu.dependencies)[1] 
        J .+= num_states
        if updateType === :dependent
            dependencyTypes = [fmi2DependencyKindDependent]
        else
            dependencyTypes = [fmi2DependencyKindDependent, fmi2DependencyKindTunable, fmi2DependencyKindDiscrete]
        end
            for (i, j, v) in zip(I, J, V)
            if v ∉ dependencyTypes
                rem_edge!(graph, i, j)
            end
        end
        return graph
    else
        if updateType !== :independent
            @warn "Undefined update type"
        end
        # return empty simple graph
        return SimpleGraph(nv(graph))
    end 
end
function updateGraph(dependencies::AbstractMatrix{fmi2DependencyKind}; updateType::Symbol) ::SimpleGraph
    graph = SimpleGraph(length(dependencies))
    
    I, J, V = findnz(dependencies)
    # add offset to J vertex
    num_states = size(dependencies)[1] 
    J .+= num_states

    dependencyTypes = [fmi2DependencyKindDependent]
    if updateType ∈ [:all, :constant, :fixed, :tunable, :discrete]
        dependencyTypes.append!([fmi2DependencyKindTunable, fmi2DependencyKindDiscrete])
    end
    if updateType ∈ [:all, :constant, :fixed]
        dependencyTypes.append!([fmi2DependencyKindConstant, fmi2DependencyKindFixed])
    end
        
    for (v1, v2, value) in zip(I, J, V)
        if value ∈ dependencyTypes 
            add_edge!(graph, v1, v2)
        end
    end
    return graph
end
