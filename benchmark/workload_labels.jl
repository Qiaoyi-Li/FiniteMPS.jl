module WorkloadLabels

export workload_title, workload_description, workload_facts

function workload_title(case)
    p = case["parameters"]
    haskey(p,"operation") || return case["case_id"]
    return "$(p["model_name"]) · $(p["operation"]) · D=$(p["nominal_D"])"
end
workload_description(case) = get(case, "description", "")
workload_facts(case) = Pair{String,Any}[
    string(key) => value for (key, value) in sort!(collect(case["parameters"]); by=first)
    if key != "execution"
]

end
