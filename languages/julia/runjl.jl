using PyCall

np = pyimport("numpy")
ultranest = pyimport("ultranest")

function mytransform(cube)
    cube * 2 .- 1
end

function mylikelihood(params)
    n, d = size(params)
    centers = 0.1 * reshape(np.arange(d), (1, d))
    -0.5 * dropdims(sum(((params .- centers) / 0.01) .^ 2, dims=2), dims=2)
end

paramnames = ["a", "b", "c"]
sampler = ultranest.ReactiveNestedSampler(paramnames, mylikelihood, transform=mytransform, vectorized=true)
results = sampler.run()
print("result has these keys:", keys(results), "\n")

sampler.print_results()
sampler.plot()
