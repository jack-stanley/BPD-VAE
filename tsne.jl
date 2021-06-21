using TSne
using Plots
using CSV
using UMAP
using ManifoldLearning
using DecisionTree

## Helpers
rescale(A; dims=1) = (A .- mean(A, dims=dims)) ./ max.(std(A, dims=dims), eps())

## Load data
df_met = CSV.read("BPD_CLIN.csv")
x = convert(Array, df_met)[:,2:end]

class_df_BPD = CSV.read("BPD_labels.csv")
class_BPD = convert(Array, class_df_BPD)

disease_df = CSV.read("BPD_CLIN_LAB.csv")
disease_BPD = convert(Array, disease_df)

## More data
df_metA = CSV.read("BPD_A.csv")
df_metD = CSV.read("BPD_D.csv")
df_metB = CSV.read("BPD_B.csv")
A = convert(Array, df_metA)[:,2:end]'
D = convert(Array, df_metD)[:,2:end]'
B = convert(Array, df_metB)[:,2:end]'
x = [A D B]
t_x = x./maximum(x, dims=2)
class_df_BPDA = CSV.read("BPD_A_ID.csv")
class_df_BPDD = CSV.read("BPD_D_ID.csv")
class_df_BPDB = CSV.read("BPD_B_ID.csv")
class_BPDA = convert(Array, class_df_BPDA)
class_BPDD = convert(Array, class_df_BPDD)
class_BPDB = convert(Array, class_df_BPDB)
class_BPD = [class_BPDA; class_BPDD; class_BPDB]

## Fit TSne
TS = tsne(x', 2, 0, 5000, 35)

## Fit UMAP
UM = umap(Float32.((t_x)),2,n_neighbors=15)

## Fit Random Forest
model = build_forest(string.(disease_BPD[:,2]), float.(x), 4, 100, 0.7)
apply_forest(model, float.(x))
accuracy = nfoldCV_forest(string.(disease_BPD[:,2]), float.(x), 3, 4)

UM[1,:]
group=class_BPD[:,2]
scatter(UM[1,:],UM[2,:], group=class_BPD[:,2], title="UMAP Disease Clusters")
scatter(UM[1,:],UM[2,:], group=disease_BPD[:,2])

savefig("UMAP_ABD_Disease")
