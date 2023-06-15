import * as React from 'react';
import { useDispatch, useSelector } from 'react-redux';
import Dialog from '@mui/material/Dialog';
import DialogTitle from '@mui/material/DialogTitle';
import DialogContent from '@mui/material/DialogContent';
import DialogActions from '@mui/material/DialogActions';
import Stack from '@mui/material/Stack';
import Slider from '@mui/material/Slider';
import { OPTION_CHANGE } from '../../store/actions';
import Button from '@mui/material/Button';
import Typography from '@mui/material/Typography';


export default function Option() {
    const { l_h, l_s, l_v, u_h, u_s, u_v, open } = useSelector((state) => state.option)
    function valuetext(value) {
        return `${value}`;
    }
    const dispatch = useDispatch();

    const handleClose = () => {
        dispatch({
            type: OPTION_CHANGE,
            open: false
        })
    };
    const handleChangeOption = (event, newValue) => {
        dispatch({
            type: OPTION_CHANGE,
            [event.target.name]: newValue
        })
    };

    return (
        <>
            <Dialog open={open} onClose={handleClose}>
                <DialogTitle>Option</DialogTitle>
                <DialogContent>
                    <Stack spacing={2} direction="row" sx={{ height: '60px', width: '500px' }} alignItems="flex-end">
                        <Typography id="input-slider" gutterBottom>
                            L_H
                        </Typography>
                        <Slider
                            name='l_h'
                            aria-label="L-H"
                            defaultValue={2}
                            size="small"
                            getAriaValueText={valuetext}
                            step={1}
                            valueLabelDisplay="auto"
                            min={0}
                            max={179}
                            value={l_h}
                            onChange={handleChangeOption}
                        />
                        <Typography id="input-slider" gutterBottom>
                            L_S
                        </Typography>
                        <Slider
                            name='l_s'
                            aria-label="L-S"
                            defaultValue={2}
                            size="small"
                            getAriaValueText={valuetext}
                            step={1}
                            valueLabelDisplay="auto"
                            min={0}
                            max={255}
                            value={l_s}
                            onChange={handleChangeOption}
                        />
                        <Typography id="input-slider" gutterBottom>
                            L_V
                        </Typography>
                        <Slider
                            name='l_v'
                            aria-label="L-V"
                            defaultValue={2}
                            size="small"
                            getAriaValueText={valuetext}
                            step={1}
                            valueLabelDisplay="auto"
                            min={0}
                            max={255}
                            value={l_v}
                            onChange={handleChangeOption}
                        />
                    </Stack>
                    <Stack spacing={2} direction="row" sx={{ height: '60px', width: '500px' }} alignItems="flex-end">
                        <Typography id="input-slider" gutterBottom>
                            U_H
                        </Typography>
                        <Slider
                            name='u_h'
                            aria-label="U-H"
                            defaultValue={2}
                            size="small"
                            getAriaValueText={valuetext}
                            step={1}
                            valueLabelDisplay="auto"
                            min={0}
                            max={179}
                            value={u_h}
                            onChange={handleChangeOption}
                        />
                        <Typography id="input-slider" gutterBottom>
                            U_S
                        </Typography>
                        <Slider
                            name='u_s'
                            aria-label="U-S"
                            defaultValue={2}
                            size="small"
                            getAriaValueText={valuetext}
                            step={1}
                            valueLabelDisplay="auto"
                            min={0}
                            max={255}
                            value={u_s}
                            onChange={handleChangeOption}
                        />
                        <Typography id="input-slider" gutterBottom>
                            U_V
                        </Typography>
                        <Slider
                            name='u_v'
                            aria-label="U-V"
                            defaultValue={2}
                            size="small"
                            getAriaValueText={valuetext}
                            step={1}
                            valueLabelDisplay="auto"
                            min={0}
                            max={255}
                            value={u_v}
                            onChange={handleChangeOption}
                        />
                    </Stack>
                </DialogContent>

                <DialogActions>
                    <Button onClick={handleClose}>Close</Button>
                </DialogActions>
            </Dialog>
        </>
    )
}