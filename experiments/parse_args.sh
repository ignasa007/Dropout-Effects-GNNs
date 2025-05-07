read_list_arg() {
    local __varname=$1
    shift
    local values=()
    while [[ $# -gt 0 && "$1" != --* ]]; do
        values+=("$1")
        shift
    done
    eval "$__varname=(\"\${values[@]}\")"
    remaining_args=("$@")  # Store remaining args in a global temp var
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --datasets)
                shift; read_list_arg datasets "$@"; set -- "${remaining_args[@]}"
                ;;
            --gnns)
                shift; read_list_arg gnns "$@"; set -- "${remaining_args[@]}"
                ;;
            --dropouts)
                shift; read_list_arg dropouts "$@"; set -- "${remaining_args[@]}"
                ;;
            --drop_ps)
                shift; read_list_arg drop_ps "$@"; set -- "${remaining_args[@]}"
                ;;
            --info_save_ratios)
                shift; read_list_arg info_save_ratios "$@"; set -- "${remaining_args[@]}"
                ;;
            --hidden_size|--depth|--attention_heads|--pooler|--learning_rate|--weight_decay|--n_epochs|--device_index|--total_samples)
                var="${1/--/}"; eval "$var=\"$2\""; shift 2
                ;;
            --bias)
                if [[ $# -gt 1 && "$2" != --* ]]; then
                    case "$2" in 
                        true|false)
                            bias="$2"; shift 2
                            ;;
                        *)
                            echo "Error: --bias accepts only 'true' or 'false' as an optional value."
                            exit 1
                            ;;
                    esac
                else
                    bias=true; shift
                fi
                ;;
            --no_bias)
                if [[ $# -gt 1 && "$2" != --* ]]; then
                    echo "Error: --no_bias does not take a value. Use just --no_bias."
                    exit 1
                fi
                bias=false; shift
                ;;
            *)
                echo "Unknown argument: $1"
                exit 1
                ;;
        esac
    done
}