import { OPTION_CHANGE } from './actions';

const initialState = {
    l_h: 0,
    l_s: 21,
    l_v: 2,
    u_h: 179,
    u_s: 255,
    u_v: 255,
    open: false
};

const optionReducer = (state = initialState, action) => {
    switch (action.type) {
        case OPTION_CHANGE:
            return {
                ...state,
                l_h: action.l_h ?? state.l_h,
                l_s: action.l_s ?? state.l_s,
                l_v: action.l_v ?? state.l_v,
                u_h: action.u_h ?? state.u_h,
                u_s: action.u_s ?? state.u_s,
                u_v: action.u_v ?? state.u_v,
                open: action.open ?? state.open
            }
        default:
            return state;
    }
}

export default optionReducer;