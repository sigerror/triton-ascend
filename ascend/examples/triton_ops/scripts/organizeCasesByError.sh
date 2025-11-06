#!/bin/bash

# Init variables
current_test=""
declare -A error_map
encounter_submit_bugreport=0
encounter_run_on_operation=0
compile_error="Compilation Error But without meaningful message"
# Process input files
while IFS= read -r line; do
    # Check if it is a test dividing line.
    if [[ "$line" =~ ^_{5,}[[:space:]]+test_[^[:space:]]+[[:space:]]+_{5,}$ ]]; then
        # Extract test case name
        current_test=$(echo "$line" | sed -E 's/^_{5,}[[:space:]]+(test_[^[:space:]]+)[[:space:]]+_{5,}$/\1/')
        continue
    fi

    # If it is an empty line or not an error line, skip it.
    [[ -z "$line" ]] && continue
    [[ "$line" != E* ]] && continue

    # Clean up the content: Remove the E prefix and loc information.
    cleaned_line=$(echo "$line" | sed -E 's/^E[[:space:]]+//' | sed -E 's/loc\([^)]*\):[[:space:]]*//g')

    # Skip blank lines or specific information lines
    [[ -z "$cleaned_line" ]] && continue

    if [[ "$encounter_submit_bugreport" == "1" ]]; then
        if [[ "$cleaned_line" == *"[ERROR][Triton][END]"* || "$cleaned_line" == *"Stack dump"* ]] && [[ "$encounter_run_on_operation" == "0" ]]; then
            # Now we passed the stack backtrace region starting from "submit a bug report"
            cleaned_line="$compile_error"
            error_map["$cleaned_line"]+=" $current_test"
            encounter_submit_bugreport=0
            continue
        fi
        # we select the core line in the stack backtrace region starting from "submit a bug report"
        if [[ "$cleaned_line" == *"::runOnOperation()"* ]]; then
            if [[ -n "$current_test" ]]; then
                cleaned_line=$(echo $cleaned_line | sed -E 's/ ([0-9a-fx]+) / /; s/(runOnOperation\(\)).*/\1/')
                error_map["$cleaned_line"]+=" $current_test"
                encounter_run_on_operation=1
                continue
            fi
        fi
        if [[ "$cleaned_line" == *"[ERROR][Triton][END]"* ]]; then
            # Now we passed the stack backtrace region starting from "submit a bug report"
            encounter_submit_bugreport=0
            encounter_run_on_operation=0
        fi
        continue
    fi

    [[ "$cleaned_line" == *"///------------------[ERROR][Triton][BEG]------------------"* ]] && continue
    [[ "$cleaned_line" == *"///------------------[ERROR][Triton][END]------------------"* ]] && continue
    [[ "$cleaned_line" == *"The compiled kernel cache is in"* ]] && continue
    [[ "$cleaned_line" == *"Failed to run BiShengIR pipeline"* ]] && continue
    [[ "$cleaned_line" == *"encounters error:"* ]] && continue

    [[ "$cleaned_line" == *"triton.compiler.errors.MLIRCompilationError"* ]] && continue
    [[ "$cleaned_line" == *"error: Failed to run BiShengHIR pipeline"* ]] && continue

    [[ "$cleaned_line" == *"error executing"* ]] && continue
    [[ "$cleaned_line" == *">>>"* ]] && continue
    [[ "$cleaned_line" == *"Failed to compile BiShengLIR for device"* ]] && continue
    [[ "$cleaned_line" == *"-m aicorelinux -Ttext"* ]] && continue

    # accuracy error: keeps details of int
    [[ "$cleaned_line" == *"<built-in method"* ]] && continue
    # [[ "$cleaned_line" == *"AssertionError: assert False"* ]] && continue
    [[ "$cleaned_line" == *"assert torch.equal"* ]] && continue

    [[ "$cleaned_line" == "E" ]] && continue

    # accuracy error: we have only singe line for float
    # [[ "$cleaned_line" == *"AssertionError: Tensor-likes are not close"* ]] && continue
    [[ "$cleaned_line" == *"torch.testing.assert_close"* ]] && continue
    [[ "$cleaned_line" == *"Mismatched elements:"* ]] && continue
    [[ "$cleaned_line" == *"Greatest absolute difference:"* ]] && continue
    [[ "$cleaned_line" == *"Greatest relative difference:"* ]] && continue

    # compilation error: we keep the message "error in backend: Cannot select: intrinsic"
    [[ "$cleaned_line" == *"clang frontend command failed with exit code"* ]] && continue
    [[ "$cleaned_line" == *"clang version"* ]] && continue
    [[ "$cleaned_line" == *"Target: aarch64-unknown-linux-gnu"* ]] && continue
    [[ "$cleaned_line" == *"Thread model: posix"* ]] && continue
    [[ "$cleaned_line" == *"InstalledDir:"* ]] && continue
    [[ "$cleaned_line" == *"--cce-aicore-only -O2 -cce-bitcode-is-aicore"* ]] && continue

    # compilation error, we keep the message "unsupported bit width"
    [[ "$cleaned_line" == *"UNREACHABLE executed at"* ]] && continue
    if [[ "$cleaned_line" == *"PLEASE submit a bug report"* ]]; then
        encounter_submit_bugreport=1
        continue
    fi
    # [[ "$cleaned_line" == *"PLEASE submit a bug report"* ]] && continue
    [[ "$cleaned_line" == *"Program arguments"* ]] && continue
    [[ "$cleaned_line" == *"Stack dump"* ]] && continue
    # [[ "$cleaned_line" == *"bishengir-compile"* ]] && continue
    [[ "$cleaned_line" == *"triton.compiler.errors.CompilationError"* ]] && continue

    [[ "$cleaned_line" == *"Vector with padded layout"* ]] && continue
    [[ "$cleaned_line" == ", but got"* ]] && continue
    [[ "$cleaned_line" == " __main__ - INFO"* ]] && continue
    [[ "$cleaned_line" == "^" ]] && continue
    [[ "$cleaned_line" == *"mask should be a 1-D vector."* ]] && continue
    [[ "$cleaned_line" == *"Attempted to vectorize, but failed"* ]] && continue
    [[ "$cleaned_line" == *"Failed to vectorize the function."* ]] && continue

    if [[ $cleaned_line == "ld.lld: error: undefined symbol:"* ]]; then
        cleaned_line="ld.lld: error: undefined symbol: _mlir_ciface_"
    fi
    if [[ $cleaned_line == *"must be i1 type, less than 256 bits of 1-bit signless integer values or AVE Hardware mask type, but got"* ]]; then
        cleaned_line="must be i1 type, less than 256 bits of 1-bit signless integer values or AVE Hardware mask type, but got"
    fi
    # Add the current test to the corresponding error map.
    if [[ -n "$current_test" ]]; then
        error_map["$cleaned_line"]+=" $current_test"
    fi
done < "$1"

declare -A other_tests
for error_type in "${!error_map[@]}"; do
    if [[ "$error_type" != "$compile_error" ]]; then
        for test in ${error_map[$error_type]}; do
            other_tests["$test"]=1
        done
    fi
done

# Handle "compile_error" type, remove use cases that duplicate other error types.
if [[ -n "${error_map["$compile_error"]}" ]]; then
    new_comm_tests=""
    # Iterate through all test cases under compile_error.
    for test in ${error_map["$compile_error"]}; do
        # If the test case does not fall under other error categories, it should be retained.
        if [[ -z "${other_tests[$test]}" ]]; then
            if [[ -n "$new_comm_tests" ]]; then
                new_comm_tests+=" $test"
            else
                new_comm_tests="$test"
            fi
        fi
    done
    # Update the test case list for compile_error
    if [[ -z "$new_comm_tests" ]]; then
        unset error_map["$compile_error"]
    else
        error_map["$compile_error"]="$new_comm_tests"
    fi
fi


# Output the result
for error in "${!error_map[@]}"; do
    echo "Error: $error"
    echo "Tests:"
    # Remove duplicate tests and format the output.
    echo "${error_map[$error]}" | tr ' ' '\n' | sort -u | grep -v '^$' | sed 's/^/  /'
    echo "---------------------"
done
