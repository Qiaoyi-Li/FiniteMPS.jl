module WorkloadLabels

export workload_title, workload_description, workload_facts

workload_title(case) = case["case_id"]
workload_description(case) = get(case, "description", "")
workload_facts(case) = Pair{String,Any}[
    string(key) => value for (key, value) in sort!(collect(case["parameters"]); by=first)
    if key != "execution"
]

end
