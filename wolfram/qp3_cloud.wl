(* Paste this file into a WolframCloud notebook input cell and evaluate it.
   First upload Orevkov's qp3.m from https://www.math.univ-toulouse.fr/~orevkov/qp3.m
   and the run's qp3_input.json into your WolframCloud files.  Change the
   CloudObject names below if you placed either file in a subdirectory. *)

qp3File = CloudObject["qp3.m"];
inputFile = CloudObject["qp3_input.json"];
Get[qp3File];
input = Import[inputFile, "RawJSON"];
allCases = input["cases"];

(* Run one call first to initialize the kernel and the function. *)
qp3[{2, 1, -2}];

measureCase[item_] := Module[{samples, answers, status},
  samples = Table[
    AbsoluteTiming[TimeConstrained[qp3[item["word"]], 30, $Failed]],
    {3}
  ];
  answers = samples[[All, 2]];
  status = Which[
    MemberQ[answers, $Failed], "timeout",
    Not[SameQ @@ answers], "inconsistent answers",
    True, "ok"
  ];
  <|
    "case_id" -> item["case_id"],
    "word" -> item["word"],
    "qp3" -> If[status === "ok", First[answers], Null],
    "elapsed_seconds" -> Median[samples[[All, 1]]],
    "repetitions" -> 3,
    "status" -> status
  |>
];

results = measureCase /@ allCases;
Dataset[results]
