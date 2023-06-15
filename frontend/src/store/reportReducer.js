import { REPORT_CHANGE } from './actions';

const initialState = {
    open: false,
    data: "",
    cm: ""
};

const reportReducer = (state = initialState, action) => {
    switch (action.type) {
        case REPORT_CHANGE:
            return {
                ...state,
                open: action.open ?? state.open,
                data: action.data ?? state.data,
                cm: action.cm ?? state.cm
            }
        default:
            return state;
    }
}

export default reportReducer;