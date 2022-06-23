using LightGraphs
using LightGraphs: AbstractGraph
using SparseArrays: findnz

"""
Create a SimpleGraph out of the dependency matrix.
"""
function createGraph(fmu::FMU2) ::SimpleGraph
    if !isdefined(fmu, :dependencies)
        @warn "Dependencies are not defined"
        return
    end
    I, J, V = findnz(fmu.dependencies)

    # add offset to J vertex
    num_states = size(fmu.dependencies)[1] 
    J .+= num_states

    fmu.graph  = SimpleGraph(num_states * 2)
    for edge in zip(I, J)
        v1, v2 = edge
        add_edge!(fmu.graph, v1, v2)
    end
    fmu.graph
end

function getVertices(num_vertices::Int; coloringType::fmi2Coloring)
    num_half_vertices = Integer(num_vertices/2)
    if coloringType == fmi2ColoringRows
        return 1:num_half_vertices
    elseif coloringType == fmi2ColoringColumns
        return num_half_vertices+1:num_vertices
    else
        return 1:num_vertices
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
function partialColoringD2(fmu::FMU2; coloringType::fmi2Coloring=fmi2ColoringRows)
    if !isdefined(fmu, :graph)
        createGraph(fmu)
    end
    if !isdefined(fmu, :colors) || fmu.colorType != coloringType
        fmu.colorType = coloringType
        fmu.colors = partialColoringD2(fmu.graph; coloringType=coloringType)
    end
    fmu.colors
end
function partialColoringD2(g::AbstractGraph; coloringType::fmi2Coloring)
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

  ==> slow but smallest coloring
"""
function starColoringD2Alg1(fmu::FMU2; coloringType::fmi2Coloring=fmi2ColoringRows)
    if !isdefined(fmu, :graph)
        createGraph(fmu)
    end
    if !isdefined(fmu, :colors) || fmu.colorType != coloringType
        fmu.colorType = coloringType
        fmu.colors = starColoringD2Alg1(fmu.graph; coloringType=coloringType)
    end
    fmu.colors
end
function starColoringD2Alg1(g::AbstractGraph; coloringType::fmi2Coloring)
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
function starColoringV2Alg2(fmu::FMU2; coloringType::fmi2Coloring=fmi2ColoringRows)
    if !isdefined(fmu, :graph)
        createGraph(fmu)
    end
    if !isdefined(fmu, :colors) || fmu.colorType != coloringType
        fmu.colorType = coloringType
        fmu.colors = starColoringV2Alg2(fmu.graph; coloringType=coloringType)
    end
    fmu.colors
end
function starColoringV2Alg2(g::AbstractGraph; coloringType::fmi2Coloring)
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
function getMinColor(forbidden_colors::AbstractVector, vertex_i::Integer)
    c = 1
    while (forbidden_colors[c] == vertex_i)
        c+=1
    end
    c
end
